import "dotenv/config";
import { Pinecone } from "@pinecone-database/pinecone";

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pc.Index(process.env.PINECONE_INDEX);

const stats = await index.describeIndexStats();

console.log(stats);
