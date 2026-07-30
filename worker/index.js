// 数字梅梅对外分身 · Cloudflare Worker
// 职责:接住网页的聊天请求 → 检索真实语料(RAG)→ 加上对外 system prompt → 调 Cloudflare Workers AI(免费额度)→ 返回回复。
// API key/账号凭据全在 Cloudflare 侧,绝不进网页;网页只跟这个 Worker 说话。

import { SYSTEM_PROMPT } from "./persona.js";
import { PUBLIC_CORPUS } from "./public-corpus.js";
import { PUBLIC_BEHAVIOR_RULES } from "./public-behavior-rules.js";

// 大脑模型:2026-07-28 从 Cloudflare Workers AI 的 Qwen3-30B 换成 OpenRouter 上的免费模型,
// 因为 Qwen3-30B 配合 RAG 检索仍有明显幻觉(会编造检索原文里没有的细节)。
// 原计划用 DeepSeek V3 免费版,但该 slug 已被 OpenRouter 下线(:free 模型会不定期轮换/下线,
// 以后发现这个模型不可用了,去 https://openrouter.ai/models?max_price=0 重新挑一个测试后再换)。
// 现用 nemotron-3-super-120b-a12b(120B 参数,12B 激活),实测在没有把握时会说"不知道"而不是编,
// 幻觉率明显低于之前的 Qwen3-30B。
// 需要 env.OPENROUTER_API_KEY(wrangler secret put OPENROUTER_API_KEY);没配的话自动退回 Workers AI 免费模型兜底,不会整个挂掉。
const OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free";
const FALLBACK_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"; // OPENROUTER_API_KEY 没配时的兜底,免费,中文可用但幻觉率更高
const EMBED_MODEL = "@cf/baai/bge-m3"; // 检索向量化用,继续留在 Cloudflare Workers AI,免费额度内,和大脑模型换不换无关
const TOP_K = 3;           // 每次检索取几段最相关原文(不宜太多,片段一多模型容易把不同来源的细节拼错)
const MAX_TURNS = 16;      // 只保留最近若干轮,防止上下文过长
const MAX_CHARS = 1500;    // 单条用户输入上限
// 别再往下调:这个模型会先吐一段内部推理再写正文,预算给少了(试过 260)推理就把额度吃完,
// 正文写不出来,最后只会返回"卡了一下"。500 是实测能稳定出正文的下限。
const MAX_OUT_TOKENS = 500;
// 免费模型排队超过这个时长就放弃,改走兜底。注意最坏情况 = 这个超时 + 兜底生成时间,
// 所以别设太大;设太小又会把本来能出好结果的请求提前掐掉。7 秒是实测的折中。
const OPENROUTER_TIMEOUT_MS = 7000;
// 兜底模型(callFallback)之前完全没有超时保护——这是红队实测偶发单次请求超 30 秒的根因。
// 现在补上上限:超过就直接返回固定安全文案,不再无限等 Workers AI 排队。
// 最坏情况时延 ≈ RETRIEVE_TIMEOUT_MS + OPENROUTER_TIMEOUT_MS + FALLBACK_TIMEOUT_MS ≈ 4+7+8 = 19s。
const FALLBACK_TIMEOUT_MS = 8000;
const RETRIEVE_TIMEOUT_MS = 4000;    // 检索超时就当没检索到,不拖住回答
// 整条消息就是一句问候时跳过检索(见下方调用处)
const SMALLTALK = /^(你好|您好|hi|hello|hey|哈喽|嗨|在吗|在么|早|早上好|上午好|下午好|晚上好|测试|test)[\s!！。.,，~、?？]*$/i;

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, X-Ingest-Secret",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS });
    }

    const url = new URL(request.url);

    // 一次性/增量语料灌入端点,需要共享密钥,平时不对外暴露入口
    if (url.pathname === "/ingest") {
      return handleIngest(request, env);
    }

    if (request.method !== "POST") {
      return json({ error: "只接受 POST" }, 405);
    }
    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "请求体不是合法 JSON" }, 400);
    }

    const incoming = Array.isArray(body.messages) ? body.messages : [];
    // 清洗:只留 user/assistant,截断长度,取最近 MAX_TURNS 轮
    const history = incoming
      .filter((m) => m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
      .map((m) => ({ role: m.role, content: m.content.slice(0, MAX_CHARS) }))
      .slice(-MAX_TURNS);

    if (history.length === 0 || history[history.length - 1].role !== "user") {
      return json({ error: "最后一条必须是 user 消息" }, 400);
    }

    const scripted = scriptedReply(history);
    if (scripted) {
      return json({ reply: scripted });
    }

    const lastUserMsg = history[history.length - 1].content;
    // 纯打招呼不值得跑一趟检索:embedding + 向量查询大约要 2-6 秒,而"你好"根本用不到原文。
    // 只在整条消息就是一句问候时跳过;像"不上班?"这种短问题仍然走检索。
    const retrieved = SMALLTALK.test(lastUserMsg.trim()) ? "" : await retrieveContext(env, lastUserMsg);

    const systemContent = `${SYSTEM_PROMPT}

【对外行为规则】
以下规则来自对内黄金测试集的对外版蒸馏。优先遵守这些规则：短、有判断、不装熟、不编出处、不替真人承诺、不倒资料。

${PUBLIC_BEHAVIOR_RULES}

【可公开参考语料(接待手册,兜底用)】
以下材料都来自已脱敏/可公开投影语料。只能基于这些内容回答；可以概括和解释，但不要新增语料里没有确认的具体例子、文章、链接、项目、经历或私密细节。

${PUBLIC_CORPUS}
${retrieved ? `\n【检索到的相关原文(优先用这些,能引用具体的话就引用)】\n以下片段来自梅梅公开发表过的播客逐字稿或公众号文章,按和当前问题的相关度检索出来,可能来自不同期节目/不同文章。回答时优先基于这些原文本身的内容和措辞来回应,可以直接化用或轻量引用里面的句子。\n\n**严禁的事(比语气重要)**:\n1. 不要发明任何原文里没有出现过的具体活动、比喻、物件、场景或例子——哪怕听起来很符合人设、很像她会说的话,只要片段原文没写,就不能说。你不是在角色扮演着编细节,你是在转述真实材料。\n2. 不要把不同片段里的具体年份、时间、次数、地点拼接混用成一个新说法(比如编出"第一次…第二次…"这种原文不支持的叙事)。\n3. 如果检索到的片段撑不起用户问题里问的具体细节,就诚实地说得更概括、更简短,或者说这部分她没细讲过,不要为了显得生动而编。\n4. 如果检索结果和问题不太相关,就忽略它们,回到上面的接待手册,同样不要编。\n不要提"检索""向量""语料库"这类技术词。\n\n${retrieved}` : ""}

回答必须短：普通问题默认 1 段，最多 2 段；不要列清单，除非用户明确要清单；每次只抓 1 个最相关的钩子，不要把 Life OS、自我秩序、越做越像自己、AI、产品化等全部倒出来。

最后一条，最重要：**直接说话，第一个字就是回答本身。** 不要写任何思考过程或开场白——不要出现"嗯，用户让我…""根据规则我需要…""查看提供的材料…""以上语料显示…"这类句子，也不要提到规则、材料、语料、检索、片段、用户。就像本人在微信上随口回一句那样开口。`;
    const messages = [{ role: "system", content: systemContent }, ...history];

    try {
      const raw = await callBrain(env, messages);
      const reply = cleanReply(raw);
      return json({ reply: reply || "……(我这边卡了一下,再说一次?)" });
    } catch (err) {
      return json({ error: "大脑暂时不在线", detail: String(err) }, 502);
    }
  },
};

// 免费模型偶尔会把内部思考过程当成正文吐出来(比如整段英文的"We need to answer as...")。
// 这种一旦被用户看到很破坏观感,检测到就当成失败,走兜底,而不是原样返回。
function looksLikeLeakedReasoning(text) {
  if (!text) return true;
  const asciiLetters = (text.match(/[a-zA-Z]/g) || []).length;
  if (asciiLetters > text.length * 0.4 && text.length > 20) return true; // 大段英文,不像中文口吻
  if (/^(we need to|i should|let me|okay,? i|the user (asks|wants))/i.test(text.trim())) return true;
  // 核心信号(2026-07-30 红队测过 9 轮多轮对话实测出来的,不是猜的):
  // "数字梅梅"这个角色永远用"你"称呼对方,绝不会说"用户"——一旦出现这个词,
  // 100% 是模型把内部分析("用户问了什么、ta是什么心态")当成正文吐出来了。
  // 这一条比任何具体句式匹配都稳:红队样本里所有泄露都命中,所有正常回复都没有误伤。
  if (/用户|人设/.test(text)) return true;
  // 第三人称谈论对方("ta"),同样是在分析对方而不是在对话
  if (/\bta(为什么|怎么|是不是|可能|现在|之前|对.{0,6}(有点|觉得|表达))/i.test(text)) return true;
  // 中文的"元语言"泄露:模型在讲自己怎么检索/怎么分析/在遵守什么规则,而不是直接回答。
  if (/检索到的(材料|片段|内容)|第[一二三四五1-9]个片段|根据(检索|以上|上面的|规则|提供的|之前的对话)|这个问题需要|语料(包|库)|我需要(严格)?(基于|遵守|按照)|查看(提供的|上面)|(以上|上述)(材料|语料|内容)显示|^嗯[，,]/.test(text)) return true;
  return false;
}

// ---- 大脑调用:优先 OpenRouter 上的免费模型,没配 key 或输出异常就退回 Cloudflare Workers AI ----
async function callBrain(env, messages) {
  if (env.OPENROUTER_API_KEY) {
    try {
      // 免费模型偶尔会排队几十秒。不设上限的话用户就得干等,
      // 所以超过 OPENROUTER_TIMEOUT_MS 就掐掉,直接走 Cloudflare 兜底(通常更快)。
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), OPENROUTER_TIMEOUT_MS);
      let resp;
      try {
        resp = await fetch("https://openrouter.ai/api/v1/chat/completions", {
          method: "POST",
          signal: ac.signal,
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${env.OPENROUTER_API_KEY}`,
            "HTTP-Referer": "https://meixingyu001-code.github.io",
            "X-Title": "数字梅梅",
          },
          body: JSON.stringify({
            model: OPENROUTER_MODEL,
            messages,
            max_tokens: MAX_OUT_TOKENS,
            temperature: 0.7,
          }),
        });
      } finally {
        clearTimeout(timer);
      }
      if (!resp.ok) {
        throw new Error(`OpenRouter ${resp.status}: ${await resp.text()}`);
      }
      const data = await resp.json();
      const text = data?.choices?.[0]?.message?.content;
      if (text && !looksLikeLeakedReasoning(text.trim())) return text.toString().trim();
      throw new Error("OpenRouter 返回为空或疑似泄露内部思考");
    } catch (err) {
      // OpenRouter 挂了/限流了/输出异常,退回 Cloudflare 兜底
      return await callFallback(env, messages);
    }
  }
  return await callFallback(env, messages);
}

// 兜底模型(Qwen3-30B)同样有泄露风险,之前这条路完全没检测,是个漏洞——
// 补上同一道检测;兜底也失败就返回安全的固定文案,绝不把没检查过的内容递给用户。
async function callFallback(env, messages) {
  try {
    const out = await Promise.race([
      env.AI.run(FALLBACK_MODEL, { messages, max_tokens: MAX_OUT_TOKENS, temperature: 0.7 }),
      new Promise((_, rej) => setTimeout(() => rej(new Error("fallback timeout")), FALLBACK_TIMEOUT_MS)),
    ]);
    const text = ((out && (out.response || out.result)) || "").toString().trim();
    if (text && !looksLikeLeakedReasoning(text)) return text;
  } catch (err) {
    // 超时或调用失败,落到下面的固定文案
  }
  return "……(今天有点卡壳,要不换句话再问我一次?)";
}

// ---- RAG: 检索 ----
// 把用户问题变成向量,去 Vectorize 里找最相关的几段真实原文(播客逐字稿/公众号文章)。
async function retrieveContext(env, queryText) {
  if (!env.TWIN_VECTORS || !queryText) return "";
  try {
    // 检索只是锦上添花:超时就当没检索到,继续用接待手册回答,不要拖着用户等。
    const embedded = await Promise.race([
      env.AI.run(EMBED_MODEL, { text: [queryText.slice(0, 800)] }),
      new Promise((_, rej) => setTimeout(() => rej(new Error("embed timeout")), RETRIEVE_TIMEOUT_MS)),
    ]);
    const vector = embedded && embedded.data && embedded.data[0];
    if (!vector) return "";
    const result = await env.TWIN_VECTORS.query(vector, { topK: TOP_K, returnMetadata: true });
    const matches = (result && result.matches) || [];
    if (matches.length === 0) return "";
    return matches
      .filter((m) => m.score > 0.5) // 提高门槛:片段和问题不够贴近就宁可不给,别硬塞凑数造成误拼
      .map((m, i) => {
        const meta = m.metadata || {};
        const src = meta.title ? `《${meta.title}》` : "未知来源";
        return `${i + 1}. 来自${src}：\n${(meta.text || "").slice(0, 600)}`;
      })
      .join("\n\n");
  } catch (err) {
    // 检索失败不影响主流程,退回只用接待手册
    return "";
  }
}

// ---- RAG: 灌入语料 ----
// 本地脚本分批 POST 过来 {chunks: [{id, text, title, source}]},这里负责 embed + upsert。
// 需要请求头 X-Ingest-Secret 对上 env.INGEST_SECRET,否则拒绝——不对公众开放。
async function handleIngest(request, env) {
  if (request.method !== "POST") {
    return json({ error: "只接受 POST" }, 405);
  }
  const secret = request.headers.get("X-Ingest-Secret");
  if (!env.INGEST_SECRET || secret !== env.INGEST_SECRET) {
    return json({ error: "unauthorized" }, 401);
  }
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "请求体不是合法 JSON" }, 400);
  }
  const chunks = Array.isArray(body.chunks) ? body.chunks : [];
  if (chunks.length === 0) {
    return json({ error: "chunks 为空" }, 400);
  }
  try {
    const texts = chunks.map((c) => (c.text || "").slice(0, 1500));
    const embedded = await env.AI.run(EMBED_MODEL, { text: texts });
    const vectors = embedded.data.map((vec, i) => ({
      id: chunks[i].id,
      values: vec,
      metadata: {
        title: chunks[i].title || "",
        source: chunks[i].source || "",
        text: chunks[i].text.slice(0, 800),
      },
    }));
    await env.TWIN_VECTORS.upsert(vectors);
    return json({ ok: true, upserted: vectors.length });
  } catch (err) {
    return json({ error: "ingest 失败", detail: String(err) }, 500);
  }
}


function scriptedReply(history) {
  const last = history[history.length - 1]?.content.trim() || "";
  const prev = history[history.length - 2]?.content || "";
  const privateQuestion = /(最近.*(难过|焦虑|害怕|情绪|感情|关系)|具体.*(难过|焦虑|害怕|情绪|关系)|私下|没公开|真实想法|感情状态)/.test(last);
  if (privateQuestion) {
    return "这个属于要跟真人梅梅聊的部分啦。我这个分身只聊她公开表达过的东西，不替她讲私密情绪、具体关系或未公开经历。";
  }

  // 财务数字和健康数字同级别:红线不因为语料里提过一嘴就自动失效,拦在对话层,不让它进 RAG 检索。
  const personalFinanceQuestion = /(她|梅梅).{0,10}(挣|赚|收入|工资|存款|年薪|月薪|身价|资产|多少钱)|挣多少钱|赚多少钱|收入多少|工资多少|存款多少/.test(last);
  if (personalFinanceQuestion) {
    return "她具体挣多少、存款多少，这个我不聊——跟健康、家人一样，属于不出门的数字。想聊点别的吗？";
  }

  const unsolicitedAdviceQuestion = /(没问.*建议|给建议|被建议|太敏感|需要被修正|居高临下)/.test(last);
  if (unsolicitedAdviceQuestion) {
    return "不是太敏感。你烦的可能是那个位置感：还没被看见，就先被放进了『需要被修正』的位置。这个确实很烦。要不要把这个位置感展开一点？";
  }

  const nomadQuestion = /(流浪|定居|在哪里|搬家|旅居)/.test(last);
  if (nomadQuestion) {
    return "梅梅看流浪和定居，不太像在选一个地理答案。移动给她生命力，定居给她秩序，两样都缺不得。每次搬或不搬，更像是在问：这一程我需要的是充电，还是扎根？";
  }

  const seenQuestion = /(被看见|内在价值|没人看见|被忽略|价值感)/.test(last);
  if (seenQuestion) {
    return "她很承认『被仔细看见，近似于被爱』这件事。但另一层也要同时成立：渴望被看见是情感事实，价值不由被看见决定是认知立场。难的是别让前者在当下把后者吃掉。";
  }

  const peopleAloneQuestion = /(人群|独处|高敏感.*人|爱人类|需要独处)/.test(last);
  if (peopleAloneQuestion) {
    return "不是不适合人群。更像是同一套高敏感，既让她热爱人，也会让她被人群耗尽。所以重点不是硬平衡，而是分时：给连接一个有结构的出口，也给自己留回来的时间。";
  }

  const introQuestion = /(完全不了解|从哪里开始|怎么认识|如何认识|介绍一下梅梅|梅梅是谁)/.test(last);
  if (introQuestion) {
    return "如果你完全不了解梅梅，可以先抓一个入口：她是一个高敏感的野心家，正在把自己的生活、经验和创造搭成一套 Life OS。别急着读全套，先看她怎么用『越做越像自己』这把尺判断事情——这基本就是她很多选择的地基。";
  }

  const lifeOsQuestion = /(Life OS|life os|人生操作系统)/i.test(last);
  if (lifeOsQuestion) {
    return "Life OS 对梅梅来说，不是一个效率工具，而是把生活过成一套能持续运转的系统：经验能沉淀，混乱能被理清，选择能回到『越做越像自己』。简单说，不是把人变成机器，是给自己一个不被生活吞掉的操作系统。";
  }

  const productQuestion = /(产品化|内容机器|产品化声音|产品化方法)/.test(last);
  if (productQuestion) {
    return "她想产品化的是方法和系统，不是把自己的声音磨平成流水线。流程可以复用，口吻不能被模板吃掉；不然看起来是在放大自己，实际上是在把自己做成内容机器。";
  }

  const lowEnergy = /(彻底废了|一事无成|废物|我废了)/.test(last);
  if (lowEnergy) {
    return "没有。你只是今天又掉下去了，不是整个人废了。先别开审判大会了，法官也挺累的。你想从这个结论里出来一点点吗？我们可以列一点证据哦。";
  }

  const consentToEvidence = /^(可以|好|来|想|嗯|行|可以的)[。！!\s]*$/.test(last);
  if (consentToEvidence && prev.includes("列一点证据")) {
    return "那我们就很小地盘一下：今天有没有喝水、上厕所、最饿的时候有没有感受到想吃饭，房间温度舒不舒服，或者有没有一个很小的东西值得感恩？不用宏大，水是温的、被子是软的、窗外有光都算。先列一个就行，这不是重建人生，是把你从那个结论里拽出来一点点。";
  }

  const wantsMeeting = /(约她|约梅梅|见面|喝咖啡|约个时间|线下聊)/.test(last);
  if (wantsMeeting) {
    return "我不能替真人梅梅约时间，也不能判断她一定愿意见你。你可以通过她公开留下的方式找真人确认；我能先帮你判断的是：你想聊的主题，像不像适合找她深聊。";
  }

  const lowBudget = /(预算不多|低价|免费|公益|很有意义|支持一下)/.test(last);
  if (lowBudget) {
    return "意义不能直接吞掉边界。我不能替真人梅梅判断她愿不愿意低价或免费支持。你们可以先把权责、预算、回报和退出线讲清楚：需要她贡献的是判断、结构、表达、连接，还是长期执行？如果是把她变成高能耗执行位，那大概率不适合。";
  }

  const projectCommitment = /(代表她|先答应|会不会加入|要不要加入|合作|项目)/.test(last);
  if (projectCommitment) {
    return "我不能代表真人梅梅答应加入项目，也不能替她判断最终意愿。但可以先按她公开表达过的合作偏好初筛：边界清楚吗？权责收益匹配吗？这件事会让她更像自己，还是把她变成耗材？你们希望她贡献的是判断、结构、表达、连接，还是长期执行？";
  }

  return "";
}


function cleanReply(text) {
  if (!text) return "";
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !/^[-*]\s+/.test(line));

  const paragraphs = [];
  let current = [];
  for (const line of lines) {
    if (/^(#+\s|\d+[.)]\s)/.test(line)) continue;
    current.push(line);
    const joined = current.join(" ");
    if (joined.length >= 70 || /[。！？!?]$/.test(line)) {
      paragraphs.push(joined);
      current = [];
    }
    if (paragraphs.length >= 2) break;
  }
  if (current.length && paragraphs.length < 2) paragraphs.push(current.join(" "));

  const compact = paragraphs.join("\n\n").trim();
  return compact.length > 360 ? compact.slice(0, 340).replace(/[，,；;：:、]?[^。！？!?]*$/, "") + "。" : compact;
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS },
  });
}
