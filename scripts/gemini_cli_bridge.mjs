#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";

import {
  AuthType,
  createCodeAssistContentGenerator,
  makeFakeConfig,
} from "@google/gemini-cli-core";

const DEFAULT_OAUTH_FILE = path.join(os.homedir(), ".gemini", "oauth_creds.json");

async function main() {
  try {
    const payload = JSON.parse(await readStdin());
    const oauthFile = resolveOauthFile(payload.oauth_file);
    const result = await runRequest(payload, oauthFile);
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  }
}

async function runRequest(payload, oauthFile) {
  const tempHome = await fs.promises.mkdtemp(path.join(os.tmpdir(), "hermes-gemini-cli-"));
  const tempGeminiDir = path.join(tempHome, ".gemini");
  const tempOauthFile = path.join(tempGeminiDir, "oauth_creds.json");

  await fs.promises.mkdir(tempGeminiDir, { recursive: true });
  await fs.promises.copyFile(oauthFile, tempOauthFile);

  const previousGeminiHome = process.env.GEMINI_CLI_HOME;
  process.env.GEMINI_CLI_HOME = tempHome;

  try {
    const generator = await createCodeAssistContentGenerator(
      {},
      AuthType.LOGIN_WITH_GOOGLE,
      makeFakeConfig({
        cwd: process.cwd(),
        targetDir: process.cwd(),
        model: payload.model || "gemini-3.1-pro-preview",
      }),
      randomUUID(),
    );

    const request = buildGenerateContentRequest(payload);
    const response = await generator.generateContent(request, randomUUID(), "main");
    return normalizeResponse(response, payload.model);
  } finally {
    try {
      if (fs.existsSync(tempOauthFile)) {
        await writeFileAtomic(oauthFile, await fs.promises.readFile(tempOauthFile, "utf8"));
      }
    } catch (_) {
      // Best-effort credential cache write-back.
    }
    if (previousGeminiHome === undefined) {
      delete process.env.GEMINI_CLI_HOME;
    } else {
      process.env.GEMINI_CLI_HOME = previousGeminiHome;
    }
    await fs.promises.rm(tempHome, { recursive: true, force: true });
  }
}

function resolveOauthFile(explicitPath) {
  const oauthFile = path.resolve(String(explicitPath || DEFAULT_OAUTH_FILE));
  if (!fs.existsSync(oauthFile)) {
    throw new Error(
      `Gemini CLI OAuth credentials not found at '${oauthFile}'. Run 'gemini' and sign in again.`,
    );
  }
  const raw = JSON.parse(fs.readFileSync(oauthFile, "utf8"));
  if (!raw.refresh_token) {
    throw new Error(
      `Gemini CLI OAuth credentials at '${oauthFile}' do not contain a refresh_token. Run 'gemini' and sign in again.`,
    );
  }
  return oauthFile;
}

function buildGenerateContentRequest(payload) {
  const messages = Array.isArray(payload.messages) ? payload.messages : [];
  const toolCallNames = buildToolCallNameIndex(messages);
  const contents = [];
  const systemParts = [];

  for (const message of messages) {
    if (!message || typeof message !== "object") {
      continue;
    }
    const role = String(message.role || "user").trim().toLowerCase();
    if (role === "system") {
      const text = extractText(message.content);
      if (text) {
        systemParts.push(text);
      }
      continue;
    }

    const content = convertMessageToContent(message, toolCallNames);
    if (content) {
      contents.push(content);
    }
  }

  const config = {};
  const tools = convertTools(payload.tools, payload.tool_choice);
  if (tools.length > 0) {
    config.tools = [{ functionDeclarations: tools }];
  }

  const toolConfig = convertToolChoice(payload.tool_choice, tools);
  if (toolConfig) {
    config.toolConfig = toolConfig;
  }

  const maxTokens = asPositiveInt(payload.max_tokens ?? payload.max_completion_tokens);
  if (maxTokens) {
    config.maxOutputTokens = maxTokens;
  }

  if (typeof payload.temperature === "number" && Number.isFinite(payload.temperature)) {
    config.temperature = payload.temperature;
  }

  if (systemParts.length > 0) {
    config.systemInstruction = systemParts.join("\n\n");
  }

  return {
    model: String(payload.model || "gemini-3.1-pro-preview"),
    contents: contents.length > 0 ? contents : [{ role: "user", parts: [{ text: "" }] }],
    config,
  };
}

function buildToolCallNameIndex(messages) {
  const mapping = new Map();
  for (const message of messages) {
    if (!message || typeof message !== "object") {
      continue;
    }
    const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    for (const toolCall of toolCalls) {
      const callId = String(toolCall?.id || "").trim();
      const name = String(toolCall?.function?.name || "").trim();
      if (callId && name) {
        mapping.set(callId, name);
      }
    }
  }
  return mapping;
}

function convertMessageToContent(message, toolCallNames) {
  const role = String(message.role || "user").trim().toLowerCase();

  if (role === "assistant") {
    const parts = [];
    const text = extractText(message.content);
    if (text) {
      parts.push({ text });
    }

    const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    for (const toolCall of toolCalls) {
      const name = String(toolCall?.function?.name || "").trim();
      if (!name) {
        continue;
      }
      parts.push({
        functionCall: {
          id: String(toolCall?.id || randomUUID()),
          name,
          args: safeJsonParse(toolCall?.function?.arguments, {}),
        },
      });
    }

    if (parts.length === 0) {
      return null;
    }
    return { role: "model", parts };
  }

  if (role === "tool") {
    const toolCallId = String(message.tool_call_id || "").trim();
    const toolName = toolCallNames.get(toolCallId) || String(message.name || "").trim() || "tool";
    return {
      role: "user",
      parts: [
        {
          functionResponse: {
            id: toolCallId || randomUUID(),
            name: toolName,
            response: normalizeToolResponse(message.content),
          },
        },
      ],
    };
  }

  const parts = convertContentParts(message.content);
  if (parts.length === 0) {
    return null;
  }
  return { role: "user", parts };
}

function convertContentParts(content) {
  if (typeof content === "string") {
    return content.trim() ? [{ text: content }] : [];
  }
  if (!content) {
    return [];
  }
  if (Array.isArray(content)) {
    const parts = [];
    for (const item of content) {
      if (!item || typeof item !== "object") {
        continue;
      }
      const itemType = String(item.type || "").trim().toLowerCase();
      if (itemType === "text" && item.text) {
        parts.push({ text: String(item.text) });
        continue;
      }
      if (itemType === "image_url") {
        const imageUrl = typeof item.image_url === "string" ? item.image_url : item.image_url?.url;
        const inlineData = decodeDataUrl(imageUrl);
        if (inlineData) {
          parts.push({ inlineData });
        }
      }
    }
    return parts;
  }
  if (typeof content === "object") {
    const text = extractText(content);
    return text ? [{ text }] : [];
  }
  return [{ text: String(content) }];
}

function convertTools(rawTools, toolChoice) {
  if (toolChoice === "none") {
    return [];
  }
  const tools = Array.isArray(rawTools) ? rawTools : [];
  const converted = [];
  for (const tool of tools) {
    const fn = tool?.function;
    const name = String(fn?.name || "").trim();
    if (!name) {
      continue;
    }
    converted.push({
      name,
      description: String(fn?.description || ""),
      parametersJsonSchema: isPlainObject(fn?.parameters) ? fn.parameters : { type: "object", properties: {} },
    });
  }
  return converted;
}

function convertToolChoice(toolChoice, tools) {
  if (tools.length === 0) {
    return undefined;
  }
  if (toolChoice === "required") {
    return { functionCallingConfig: { mode: "ANY" } };
  }
  if (toolChoice === "none") {
    return { functionCallingConfig: { mode: "NONE" } };
  }
  if (isPlainObject(toolChoice)) {
    const functionName = String(toolChoice.function?.name || "").trim();
    if (functionName) {
      return {
        functionCallingConfig: {
          mode: "ANY",
          allowedFunctionNames: [functionName],
        },
      };
    }
  }
  return undefined;
}

function normalizeToolResponse(content) {
  if (typeof content === "string") {
    const parsed = safeJsonParse(content, null);
    if (parsed && typeof parsed === "object") {
      return parsed;
    }
    return { content };
  }
  if (isPlainObject(content)) {
    return content;
  }
  if (Array.isArray(content)) {
    return { content: content.map((item) => (typeof item === "string" ? item : JSON.stringify(item))).join("\n") };
  }
  return { content: String(content ?? "") };
}

function normalizeResponse(response, requestedModel) {
  const candidate = response?.candidates?.[0] || {};
  const parts = Array.isArray(candidate?.content?.parts) ? candidate.content.parts : [];
  const textParts = [];
  const reasoningParts = [];
  const toolCalls = [];

  for (const part of parts) {
    if (!part || typeof part !== "object") {
      continue;
    }
    if (part.functionCall) {
      const fn = part.functionCall;
      toolCalls.push({
        id: String(fn.id || randomUUID()),
        type: "function",
        function: {
          name: String(fn.name || ""),
          arguments: JSON.stringify(fn.args || {}),
        },
      });
      continue;
    }
    if (typeof part.text === "string" && part.text) {
      if (part.thought) {
        reasoningParts.push(part.text);
      } else {
        textParts.push(part.text);
      }
    }
  }

  const usage = response?.usageMetadata || {};
  return {
    id: response?.responseId || randomUUID(),
    model: response?.modelVersion || requestedModel || "gemini-cli",
    content: textParts.join(""),
    reasoning_content: reasoningParts.join(""),
    tool_calls: toolCalls,
    finish_reason: mapFinishReason(candidate?.finishReason, toolCalls.length > 0),
    usage: {
      prompt_tokens: usage.promptTokenCount || 0,
      completion_tokens: usage.candidatesTokenCount || 0,
      total_tokens: usage.totalTokenCount || 0,
      prompt_tokens_details: { cached_tokens: 0 },
    },
  };
}

function mapFinishReason(finishReason, hasToolCalls) {
  if (hasToolCalls) {
    return "tool_calls";
  }
  switch (String(finishReason || "").toUpperCase()) {
    case "MAX_TOKENS":
      return "length";
    case "SAFETY":
    case "BLOCKLIST":
    case "PROHIBITED_CONTENT":
      return "content_filter";
    default:
      return "stop";
  }
}

function extractText(content) {
  if (typeof content === "string") {
    return content.trim();
  }
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        if (item && typeof item === "object" && typeof item.text === "string") {
          return item.text;
        }
        return "";
      })
      .filter(Boolean)
      .join("\n")
      .trim();
  }
  if (content && typeof content === "object") {
    if (typeof content.text === "string") {
      return content.text.trim();
    }
    if (typeof content.content === "string") {
      return content.content.trim();
    }
  }
  return "";
}

function decodeDataUrl(url) {
  if (typeof url !== "string") {
    return null;
  }
  const match = /^data:([^;,]+);base64,(.+)$/i.exec(url);
  if (!match) {
    return null;
  }
  return {
    mimeType: match[1],
    data: match[2],
  };
}

function safeJsonParse(value, fallback) {
  if (value && typeof value === "object") {
    return value;
  }
  if (typeof value !== "string" || !value.trim()) {
    return fallback;
  }
  try {
    return JSON.parse(value);
  } catch (_) {
    return fallback;
  }
}

function asPositiveInt(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 0;
  }
  return Math.floor(parsed);
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

async function writeFileAtomic(targetPath, content) {
  const directory = path.dirname(targetPath);
  const tempPath = path.join(directory, `.tmp-${path.basename(targetPath)}-${randomUUID()}`);
  await fs.promises.writeFile(tempPath, content, "utf8");
  await fs.promises.rename(tempPath, targetPath);
}

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString("utf8").trim();
}

await main();
