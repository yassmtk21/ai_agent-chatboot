import { ChatOllama } from "@langchain/ollama";

const llm = new ChatOllama({
  model: "llama3.2",
  temperature: 0,
});

const response = await llm.invoke("Hello! Introduce yourself briefly.");

console.log(response.content);
