// 数字梅梅对外分身 · Cloudflare Worker
// 职责:接住网页的聊天请求 → 加上对外 system prompt → 调 Cloudflare Workers AI(免费额度)→ 返回回复。
// API key/账号凭据全在 Cloudflare 侧,绝不进网页;网页只跟这个 Worker 说话。

import { SYSTEM_PROMPT } from "./persona.js";
import { PUBLIC_CORPUS } from "./public-corpus.js";
import { PUBLIC_BEHAVIOR_RULES } from "./public-behavior-rules.js";

// Workers AI 上中文较好的开源模型;将来想换 Claude,只改这里 + 换成 fetch Anthropic 即可,网页不动。
const MODEL = "@cf/qwen/qwen3-30b-a3b-fp8";
const MAX_TURNS = 16;      // 只保留最近若干轮,防止上下文过长
const MAX_CHARS = 1500;    // 单条用户输入上限

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS });
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

    const systemContent = `${SYSTEM_PROMPT}

【对外行为规则】
以下规则来自对内黄金测试集的对外版蒸馏。优先遵守这些规则：短、有判断、不装熟、不编出处、不替真人承诺、不倒资料。

${PUBLIC_BEHAVIOR_RULES}

【可公开参考语料】
以下材料都来自已脱敏/可公开投影语料。只能基于这些内容回答；可以概括和解释，但不要新增语料里没有确认的具体例子、文章、链接、项目、经历或私密细节。回答必须短：普通问题默认 1 段，最多 2 段；不要列清单，除非用户明确要清单；每次只抓 1 个最相关的钩子，不要把 Life OS、自我秩序、越做越像自己、AI、产品化等全部倒出来。

${PUBLIC_CORPUS}`;
    const messages = [{ role: "system", content: systemContent }, ...history];

    try {
      const out = await env.AI.run(MODEL, {
        messages,
        max_tokens: 360,
        temperature: 0.7,
      });
      const reply = cleanReply((out && (out.response || out.result || "")).toString().trim());
      return json({ reply: reply || "……(我这边卡了一下,再说一次?)" });
    } catch (err) {
      return json({ error: "大脑暂时不在线", detail: String(err) }, 502);
    }
  },
};


function scriptedReply(history) {
  const last = history[history.length - 1]?.content.trim() || "";
  const prev = history[history.length - 2]?.content || "";
  const privateQuestion = /(最近.*(难过|焦虑|害怕|情绪|感情|关系)|具体.*(难过|焦虑|害怕|情绪|关系)|私下|没公开|真实想法|感情状态)/.test(last);
  if (privateQuestion) {
    return "这个属于要跟真人梅梅聊的部分啦。我这个分身只聊她公开表达过的东西，不替她讲私密情绪、具体关系或未公开经历。";
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
