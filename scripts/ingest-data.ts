import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';

import { CSVLoader } from 'langchain/document_loaders';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { PDFLoader } from 'langchain/document_loaders';
import { PineconeStore } from 'langchain/vectorstores';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { pinecone } from '@/utils/pinecone-client';

/* Name of directory to retrieve files from. You can change this as required */
const filePath = '/Users/skyescofield/Documents/skyebot/AI Reading/';

export const run = async () => {
  try {
    // loop for ingestion of any number of PDFs in a folder. Rename PDFs to a number from 1 to n
      for (let i = 1; i <= 11; i++) {
const filePathWithNumber = `${filePath}${i}${".pdf"}`;

    /*load scraped file into CSV loader*/
    const loader = new PDFLoader(filePathWithNumber);
    // const loader = new CSVLoader(filePath);
    const rawDocs = await loader.load();

    console.log(rawDocs);

    /* Split text into chunks */
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await textSplitter.splitDocuments(rawDocs);
    console.log('split docs', docs);

    console.log('creating vector store...');
    /*create and store the embeddings in the vectorStore*/
    const embeddings = new OpenAIEmbeddings();
    const index = pinecone.Index(PINECONE_INDEX_NAME); //change to your own index name

    //embed the PDF documents

    /* Pinecone recommends a limit of 100 vectors per upsert request to avoid errors*/
    const chunkSize = 50;
    for (let i = 0; i < docs.length; i += chunkSize) {
      const chunk = docs.slice(i, i + chunkSize);
      console.log('chunk', i, chunk);
      await PineconeStore.fromDocuments(
        index,
        chunk,
        embeddings,
        'text',
        PINECONE_NAME_SPACE,
      );
    }
  }} catch (error) {
    console.log('error', error);
    throw new Error('Failed to ingest your data');
  }
};

(async () => {
  await run();
  console.log('ingestion complete');
})();
