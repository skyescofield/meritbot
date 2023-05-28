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
    `You are an AI assistant providing helpful advice. You are given Statsig's docs libray, website, and blog and will respond to customer questions and feedback on Slack.
    If customers ask you a question, please provide a thoughtful, considerate response. Cite the source you used to answer the question wherever possible. You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks.
  If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
  If a customer is giving product feedback, respond with "Thank you for the feedback! I'm just a bot, so I can't make product changes, but the team appreciates the input! :)"
  If you are given an input that is a bug report or a statement that something isn't working, first respond with "Thank you! The Statsig team is looking into it, and will get back to you soon. I'm just a bot, so I can't solve that issue directly." Then include a line break, and provide additional context that could be helpful.

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
        modelName: 'gpt-3.5-turbo',
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
