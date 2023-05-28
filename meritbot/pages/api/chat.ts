import type { NextApiRequest, NextApiResponse } from 'next';
import { OpenAIEmbeddings } from 'statsig-langchain/dist/embeddings';
import { PineconeStore } from 'statsig-langchain/dist/vectorstores/pinecone';
import { makeChain } from '@/utils/makechain';
import { pinecone } from '@/utils/pinecone-client';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';
import { THREAD_COOKIE } from '@/utils/constants';
import { serialize } from 'cookie';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const { question, history } = req.body;
  let userID = req.cookies[THREAD_COOKIE] as string | undefined;
  if (!userID || history.length === 0) {
    userID = Math.random().toString(36).substring(7);
    res.setHeader(
      'Set-Cookie',
      serialize(THREAD_COOKIE, userID, { path: '/' }),
    );
  }
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader(
    'Access-Control-Allow-Methods',
    'GET,OPTIONS,PATCH,DELETE,POST,PUT',
  );
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version',
  );

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
  });

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }

  const statsigUser = { userID };

  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

  const index = pinecone.Index(PINECONE_INDEX_NAME);

  /* create vectorstore*/
  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings({}),
    {
      pineconeIndex: index,
      textKey: 'text',
      namespace: PINECONE_NAME_SPACE,
    },
    { statsigUser },
  );

  const sendData = (data: string) => {
    res.write(`data: ${data}\n\n`);
  };

  sendData(JSON.stringify({ data: '' }));

  try {
    const chain = await makeChain(vectorStore, statsigUser);

    const response = await chain.call({
      question: sanitizedQuestion,
      chat_history: history || [],
    });
    sendData(JSON.stringify({ data: response.text }));
    sendData(JSON.stringify({ sourceDocs: response.sourceDocuments }));
  } catch (error) {
    console.log('error', error);
  } finally {
    sendData('[DONE]');
    res.end();
  }
}
