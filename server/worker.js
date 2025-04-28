import { Worker } from "bullmq";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import { OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import fs from 'fs';
import path from 'path';

// First, let's get the actual embedding dimension from Ollama
async function getEmbeddingDimension() {
  try {
    console.log("Testing embedding dimension...");
    const embeddings = new OllamaEmbeddings({
      model: "nomic-embed-text",
      baseUrl: "http://localhost:11434",
    });
    
    // Generate a test embedding to check the dimension
    const testResult = await embeddings.embedQuery("test query");
    const dimension = testResult.length;
    console.log(`Detected embedding dimension: ${dimension}`);
    return dimension;
  } catch (error) {
    console.error("Error detecting embedding dimension:", error);
    // Default to 768 if we can't detect (based on the error message)
    return 768;
  }
}

const worker = new Worker(
  "file-upload-queue",
  async (job) => {
    try {
      console.log("Processing job:", job.id);
      console.log("Job data:", job.data);
      
      // Check if job.data is already an object or needs parsing
      const fileInfo = typeof job.data === 'string' ? JSON.parse(job.data) : job.data;
      
      // Get the full file path
      const filePath = fileInfo.path || path.join(fileInfo.destination, fileInfo.filename);
      
      console.log(`Loading PDF from: ${filePath}`);
      
      // Verify file exists before processing
      if (!fs.existsSync(filePath)) {
        throw new Error(`File not found at path: ${filePath}`);
      }
      
      // Initialize Ollama embeddings
      console.log("Initializing Ollama embeddings...");
      const embeddings = new OllamaEmbeddings({
        model: "nomic-embed-text",
        baseUrl: "http://localhost:11434",
      });
      
      // Get the embedding dimension
      const dimension = await getEmbeddingDimension();
      
      // Initialize Qdrant client
      console.log("Connecting to Qdrant...");
      const qdrantClient = new QdrantClient({
        url: "http://localhost:6333",
      });
      
      // Check if collection exists and create if needed
      try {
        console.log("Checking if collection exists...");
        const collections = await qdrantClient.getCollections();
        const collectionExists = collections.collections.some(collection => 
          collection.name === "pdf-docs"
        );
        
        if (collectionExists) {
          console.log("Collection exists, checking dimensions...");
          // Get collection info to check dimensions
          const collectionInfo = await qdrantClient.getCollection("pdf-docs");
          const currentDimension = collectionInfo.config.params.vectors.size;
          
          if (currentDimension !== dimension) {
            console.log(`Dimension mismatch: Collection uses ${currentDimension} but embeddings are ${dimension}`);
            console.log("Recreating collection with correct dimensions...");
            
            // Delete the existing collection
            await qdrantClient.deleteCollection("pdf-docs");
            
            // Create new collection with correct dimensions
            await qdrantClient.createCollection("pdf-docs", {
              vectors: { 
                size: dimension,
                distance: "Cosine" 
              }
            });
            console.log(`Collection recreated with dimension: ${dimension}`);
          }
        } else {
          console.log(`Creating pdf-docs collection in Qdrant with dimension: ${dimension}`);
          await qdrantClient.createCollection("pdf-docs", {
            vectors: { 
              size: dimension,
              distance: "Cosine" 
            }
          });
        }
      } catch (error) {
        console.error("Error checking/creating collection:", error);
        throw error;
      }
  
      // Load the PDF
      console.log("Loading PDF document...");
      const loader = new PDFLoader(filePath, {
        splitPages: true
      });
      const docs = await loader.load();
      console.log(`Loaded ${docs.length} pages from PDF`);
      
      // Split documents into chunks
      console.log("Splitting document into chunks...");
      const textSplitter = new CharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });
      const splitDocs = await textSplitter.splitDocuments(docs);
      console.log(`Created ${splitDocs.length} text chunks`);
      
      // Initialize Qdrant vector store
      console.log("Initializing vector store...");
      const vectorStore = new QdrantVectorStore(embeddings, {
        url: "http://localhost:6333",
        collectionName: "pdf-docs",
      });
      
      // Process in smaller batches to avoid overwhelming Ollama
      const batchSize = 50;
      console.log(`Adding documents in batches of ${batchSize}...`);
      
      for (let i = 0; i < splitDocs.length; i += batchSize) {
        const batch = splitDocs.slice(i, i + batchSize);
        console.log(`Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(splitDocs.length/batchSize)}`);
        await vectorStore.addDocuments(batch);
        // Add a small delay between batches to avoid overwhelming Ollama
        if (i + batchSize < splitDocs.length) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
      
      console.log("All documents successfully added to vector store");
      return { success: true, docCount: splitDocs.length };
    } catch (error) {
      console.error("Error processing job:", error);
      throw error; // Re-throw to mark the job as failed
    }
  },
  {
    concurrency: 1, // Process one job at a time
    connection: {
      host: "localhost",
      port: 6379,
    },
  }
);

// Log when worker is ready
worker.on('ready', () => {
  console.log('Worker is ready to process jobs');
});

// Log when worker is processing a job
worker.on('active', job => {
  console.log(`Processing job ${job.id}`);
});

// Log when worker completes a job
worker.on('completed', (job, result) => {
  console.log(`Job ${job.id} completed with result:`, result);
});

// Log any errors
worker.on('error', err => {
  console.error('Worker encountered an error', err);
});

// Log failed jobs
worker.on('failed', (job, err) => {
  console.error(`Job ${job.id} failed with error:`, err);
});

// Keep the worker running
process.on('SIGTERM', async () => {
  console.log('Shutting down worker...');
  await worker.close();
});

console.log('PDF processing worker started');