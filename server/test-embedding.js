import "dotenv/config";
import { OllamaEmbeddings } from "@langchain/ollama";

const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
  baseUrl: "http://localhost:11434",
});

const result = await embeddings.embedQuery("Hello, this is a test.");

console.log("Embedding dimension:", result.length);
console.log("First 5 values:", result.slice(0, 5));
