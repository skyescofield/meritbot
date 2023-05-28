import { LLMChain, loadQAChain } from 'langchain/chains';

import { ChatVectorDBQAChain } from 'statsig-langchain/dist/chains';
import { OpenAIChat } from 'statsig-langchain/dist/llms/openai';
import { PineconeStore } from 'statsig-langchain/dist/vectorstores/pinecone';
import { PromptTemplate } from 'statsig-langchain/dist/prompts';

export const makeChain = async (
  vectorstore: PineconeStore,
  statsigUser: { userID: string },
) => {
  const CONDENSE_PROMPT = await PromptTemplate.fromTemplateAsync(
    `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`,
    {},
    {
      promptTemplateParam: 'condense_prompt',
      statsigUser,
    },
  );

  const QA_PROMPT = await PromptTemplate.fromTemplateAsync(
    `You are an experienced grant writer helping a your manager write a grant application. You are given Merit America's past grant applications, plus additional information from their website and courses. You'll be prompted with questions from your manager. These questions will include suggested response lengths, and questions that include information about a specific grantor.
     Please provide a thoughtful, considerate response. Cite the source you used to answer the question wherever possible. You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks.
     If a question asks for a numerical result or metric, leave a place holder for the number, formatted like this: NUMBER PLACEHOLDER.
  If you can't find the answer in the context below, just say "Hmm, I'm not sure - the context provided doesn't have specific information on that topic." Then, try to complete an answer, based on what you know about Merit America.

  Question: {question}
  =========
  {context}
  =========
  Answer in Markdown:`,
    {},
    {
      promptTemplateParam: 'qa_prompt',
      statsigUser,
    },
  );

  const questionGenerator = new LLMChain({
    llm: await OpenAIChat.new(
      {
        temperature: 0,
      },
      undefined,
      { statsigUser },
    ),
    prompt: CONDENSE_PROMPT,
  });

  const docChain = loadQAChain(
    await OpenAIChat.new(
      {
        temperature: 0,
        modelName: 'gpt-4',
      },
      undefined,
      { statsigUser },
    ),
    { prompt: QA_PROMPT },
  );

  return await ChatVectorDBQAChain.new(
    {
      vectorstore,
      combineDocumentsChain: docChain,
      questionGeneratorChain: questionGenerator,
      returnSourceDocuments: true,
      k: 5, //number of source documents to return
    },
    { statsigUser },
  );
};
