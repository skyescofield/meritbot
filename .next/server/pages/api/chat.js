"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
(() => {
var exports = {};
exports.id = "pages/api/chat";
exports.ids = ["pages/api/chat"];
exports.modules = {

/***/ "@pinecone-database/pinecone":
/*!**********************************************!*\
  !*** external "@pinecone-database/pinecone" ***!
  \**********************************************/
/***/ ((module) => {

module.exports = require("@pinecone-database/pinecone");

/***/ }),

/***/ "cookie":
/*!*************************!*\
  !*** external "cookie" ***!
  \*************************/
/***/ ((module) => {

module.exports = require("cookie");

/***/ }),

/***/ "statsig-langchain/dist/chains":
/*!************************************************!*\
  !*** external "statsig-langchain/dist/chains" ***!
  \************************************************/
/***/ ((module) => {

module.exports = require("statsig-langchain/dist/chains");

/***/ }),

/***/ "statsig-langchain/dist/embeddings":
/*!****************************************************!*\
  !*** external "statsig-langchain/dist/embeddings" ***!
  \****************************************************/
/***/ ((module) => {

module.exports = require("statsig-langchain/dist/embeddings");

/***/ }),

/***/ "statsig-langchain/dist/llms/openai":
/*!*****************************************************!*\
  !*** external "statsig-langchain/dist/llms/openai" ***!
  \*****************************************************/
/***/ ((module) => {

module.exports = require("statsig-langchain/dist/llms/openai");

/***/ }),

/***/ "statsig-langchain/dist/prompts":
/*!*************************************************!*\
  !*** external "statsig-langchain/dist/prompts" ***!
  \*************************************************/
/***/ ((module) => {

module.exports = require("statsig-langchain/dist/prompts");

/***/ }),

/***/ "statsig-langchain/dist/vectorstores/pinecone":
/*!***************************************************************!*\
  !*** external "statsig-langchain/dist/vectorstores/pinecone" ***!
  \***************************************************************/
/***/ ((module) => {

module.exports = require("statsig-langchain/dist/vectorstores/pinecone");

/***/ }),

/***/ "langchain/chains":
/*!***********************************!*\
  !*** external "langchain/chains" ***!
  \***********************************/
/***/ ((module) => {

module.exports = import("langchain/chains");;

/***/ }),

/***/ "(api)/./config/pinecone.ts":
/*!****************************!*\
  !*** ./config/pinecone.ts ***!
  \****************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"PINECONE_INDEX_NAME\": () => (/* binding */ PINECONE_INDEX_NAME),\n/* harmony export */   \"PINECONE_NAME_SPACE\": () => (/* binding */ PINECONE_NAME_SPACE)\n/* harmony export */ });\n/**\n * Change the index and namespace to your own\n */ const PINECONE_INDEX_NAME = \"meritdocs\";\nconst PINECONE_NAME_SPACE = \"\"; //namespace is optional for your vectors\n\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGFwaSkvLi9jb25maWcvcGluZWNvbmUudHMuanMiLCJtYXBwaW5ncyI6Ijs7Ozs7QUFBQTs7Q0FFQyxHQUVELE1BQU1BLHNCQUFzQjtBQUU1QixNQUFNQyxzQkFBc0IsSUFBSSx3Q0FBd0M7QUFFcEIiLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9ncHQ0LWxhbmdjaGFpbi1wZGYtY2hhdGJvdC8uL2NvbmZpZy9waW5lY29uZS50cz82ZjA5Il0sInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQ2hhbmdlIHRoZSBpbmRleCBhbmQgbmFtZXNwYWNlIHRvIHlvdXIgb3duXG4gKi9cblxuY29uc3QgUElORUNPTkVfSU5ERVhfTkFNRSA9ICdtZXJpdGRvY3MnO1xuXG5jb25zdCBQSU5FQ09ORV9OQU1FX1NQQUNFID0gJyc7IC8vbmFtZXNwYWNlIGlzIG9wdGlvbmFsIGZvciB5b3VyIHZlY3RvcnNcblxuZXhwb3J0IHsgUElORUNPTkVfSU5ERVhfTkFNRSwgUElORUNPTkVfTkFNRV9TUEFDRSB9O1xuIl0sIm5hbWVzIjpbIlBJTkVDT05FX0lOREVYX05BTUUiLCJQSU5FQ09ORV9OQU1FX1NQQUNFIl0sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///(api)/./config/pinecone.ts\n");

/***/ }),

/***/ "(api)/./pages/api/chat.ts":
/*!***************************!*\
  !*** ./pages/api/chat.ts ***!
  \***************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.a(module, async (__webpack_handle_async_dependencies__, __webpack_async_result__) => { try {\n__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"default\": () => (/* binding */ handler)\n/* harmony export */ });\n/* harmony import */ var statsig_langchain_dist_embeddings__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! statsig-langchain/dist/embeddings */ \"statsig-langchain/dist/embeddings\");\n/* harmony import */ var statsig_langchain_dist_embeddings__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(statsig_langchain_dist_embeddings__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var statsig_langchain_dist_vectorstores_pinecone__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! statsig-langchain/dist/vectorstores/pinecone */ \"statsig-langchain/dist/vectorstores/pinecone\");\n/* harmony import */ var statsig_langchain_dist_vectorstores_pinecone__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(statsig_langchain_dist_vectorstores_pinecone__WEBPACK_IMPORTED_MODULE_1__);\n/* harmony import */ var _utils_makechain__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @/utils/makechain */ \"(api)/./utils/makechain.ts\");\n/* harmony import */ var _utils_pinecone_client__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @/utils/pinecone-client */ \"(api)/./utils/pinecone-client.ts\");\n/* harmony import */ var _config_pinecone__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @/config/pinecone */ \"(api)/./config/pinecone.ts\");\n/* harmony import */ var _utils_constants__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! @/utils/constants */ \"(api)/./utils/constants.ts\");\n/* harmony import */ var cookie__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! cookie */ \"cookie\");\n/* harmony import */ var cookie__WEBPACK_IMPORTED_MODULE_6___default = /*#__PURE__*/__webpack_require__.n(cookie__WEBPACK_IMPORTED_MODULE_6__);\nvar __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([_utils_makechain__WEBPACK_IMPORTED_MODULE_2__, _utils_pinecone_client__WEBPACK_IMPORTED_MODULE_3__]);\n([_utils_makechain__WEBPACK_IMPORTED_MODULE_2__, _utils_pinecone_client__WEBPACK_IMPORTED_MODULE_3__] = __webpack_async_dependencies__.then ? (await __webpack_async_dependencies__)() : __webpack_async_dependencies__);\n\n\n\n\n\n\n\nasync function handler(req, res) {\n    const { question , history  } = req.body;\n    let userID = req.cookies[_utils_constants__WEBPACK_IMPORTED_MODULE_5__.THREAD_COOKIE];\n    if (!userID || history.length === 0) {\n        userID = Math.random().toString(36).substring(7);\n        res.setHeader(\"Set-Cookie\", (0,cookie__WEBPACK_IMPORTED_MODULE_6__.serialize)(_utils_constants__WEBPACK_IMPORTED_MODULE_5__.THREAD_COOKIE, userID, {\n            path: \"/\"\n        }));\n    }\n    res.setHeader(\"Access-Control-Allow-Credentials\", \"true\");\n    res.setHeader(\"Access-Control-Allow-Origin\", \"*\");\n    res.setHeader(\"Access-Control-Allow-Methods\", \"GET,OPTIONS,PATCH,DELETE,POST,PUT\");\n    res.setHeader(\"Access-Control-Allow-Headers\", \"X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version\");\n    res.writeHead(200, {\n        \"Content-Type\": \"text/event-stream\",\n        \"Cache-Control\": \"no-cache, no-transform\",\n        Connection: \"keep-alive\"\n    });\n    if (!question) {\n        return res.status(400).json({\n            message: \"No question in the request\"\n        });\n    }\n    const statsigUser = {\n        userID\n    };\n    // OpenAI recommends replacing newlines with spaces for best results\n    const sanitizedQuestion = question.trim().replaceAll(\"\\n\", \" \");\n    const index = _utils_pinecone_client__WEBPACK_IMPORTED_MODULE_3__.pinecone.Index(_config_pinecone__WEBPACK_IMPORTED_MODULE_4__.PINECONE_INDEX_NAME);\n    /* create vectorstore*/ const vectorStore = await statsig_langchain_dist_vectorstores_pinecone__WEBPACK_IMPORTED_MODULE_1__.PineconeStore.fromExistingIndex(new statsig_langchain_dist_embeddings__WEBPACK_IMPORTED_MODULE_0__.OpenAIEmbeddings({}), {\n        pineconeIndex: index,\n        textKey: \"text\",\n        namespace: _config_pinecone__WEBPACK_IMPORTED_MODULE_4__.PINECONE_NAME_SPACE\n    }, {\n        statsigUser\n    });\n    const sendData = (data)=>{\n        res.write(`data: ${data}\\n\\n`);\n    };\n    sendData(JSON.stringify({\n        data: \"\"\n    }));\n    try {\n        const chain = await (0,_utils_makechain__WEBPACK_IMPORTED_MODULE_2__.makeChain)(vectorStore, statsigUser);\n        const response = await chain.call({\n            question: sanitizedQuestion,\n            chat_history: history || []\n        });\n        sendData(JSON.stringify({\n            data: response.text\n        }));\n        sendData(JSON.stringify({\n            sourceDocs: response.sourceDocuments\n        }));\n    } catch (error) {\n        console.log(\"error\", error);\n    } finally{\n        sendData(\"[DONE]\");\n        res.end();\n    }\n}\n\n__webpack_async_result__();\n} catch(e) { __webpack_async_result__(e); } });//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGFwaSkvLi9wYWdlcy9hcGkvY2hhdC50cy5qcyIsIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7OztBQUNxRTtBQUNRO0FBQy9CO0FBQ0s7QUFDMEI7QUFDM0I7QUFDZjtBQUVwQixlQUFlUSxRQUM1QkMsR0FBbUIsRUFDbkJDLEdBQW9CLEVBQ3BCO0lBQ0EsTUFBTSxFQUFFQyxTQUFRLEVBQUVDLFFBQU8sRUFBRSxHQUFHSCxJQUFJSSxJQUFJO0lBQ3RDLElBQUlDLFNBQVNMLElBQUlNLE9BQU8sQ0FBQ1QsMkRBQWFBLENBQUM7SUFDdkMsSUFBSSxDQUFDUSxVQUFVRixRQUFRSSxNQUFNLEtBQUssR0FBRztRQUNuQ0YsU0FBU0csS0FBS0MsTUFBTSxHQUFHQyxRQUFRLENBQUMsSUFBSUMsU0FBUyxDQUFDO1FBQzlDVixJQUFJVyxTQUFTLENBQ1gsY0FDQWQsaURBQVNBLENBQUNELDJEQUFhQSxFQUFFUSxRQUFRO1lBQUVRLE1BQU07UUFBSTtJQUVqRCxDQUFDO0lBQ0RaLElBQUlXLFNBQVMsQ0FBQyxvQ0FBb0M7SUFDbERYLElBQUlXLFNBQVMsQ0FBQywrQkFBK0I7SUFDN0NYLElBQUlXLFNBQVMsQ0FDWCxnQ0FDQTtJQUVGWCxJQUFJVyxTQUFTLENBQ1gsZ0NBQ0E7SUFHRlgsSUFBSWEsU0FBUyxDQUFDLEtBQUs7UUFDakIsZ0JBQWdCO1FBQ2hCLGlCQUFpQjtRQUNqQkMsWUFBWTtJQUNkO0lBRUEsSUFBSSxDQUFDYixVQUFVO1FBQ2IsT0FBT0QsSUFBSWUsTUFBTSxDQUFDLEtBQUtDLElBQUksQ0FBQztZQUFFQyxTQUFTO1FBQTZCO0lBQ3RFLENBQUM7SUFFRCxNQUFNQyxjQUFjO1FBQUVkO0lBQU87SUFFN0Isb0VBQW9FO0lBQ3BFLE1BQU1lLG9CQUFvQmxCLFNBQVNtQixJQUFJLEdBQUdDLFVBQVUsQ0FBQyxNQUFNO0lBRTNELE1BQU1DLFFBQVE3QixrRUFBYyxDQUFDQyxpRUFBbUJBO0lBRWhELHFCQUFxQixHQUNyQixNQUFNOEIsY0FBYyxNQUFNakMseUdBQStCLENBQ3ZELElBQUlELCtFQUFnQkEsQ0FBQyxDQUFDLElBQ3RCO1FBQ0VvQyxlQUFlSjtRQUNmSyxTQUFTO1FBQ1RDLFdBQVdqQyxpRUFBbUJBO0lBQ2hDLEdBQ0E7UUFBRXVCO0lBQVk7SUFHaEIsTUFBTVcsV0FBVyxDQUFDQyxPQUFpQjtRQUNqQzlCLElBQUkrQixLQUFLLENBQUMsQ0FBQyxNQUFNLEVBQUVELEtBQUssSUFBSSxDQUFDO0lBQy9CO0lBRUFELFNBQVNHLEtBQUtDLFNBQVMsQ0FBQztRQUFFSCxNQUFNO0lBQUc7SUFFbkMsSUFBSTtRQUNGLE1BQU1JLFFBQVEsTUFBTTFDLDJEQUFTQSxDQUFDZ0MsYUFBYU47UUFFM0MsTUFBTWlCLFdBQVcsTUFBTUQsTUFBTUUsSUFBSSxDQUFDO1lBQ2hDbkMsVUFBVWtCO1lBQ1ZrQixjQUFjbkMsV0FBVyxFQUFFO1FBQzdCO1FBQ0EyQixTQUFTRyxLQUFLQyxTQUFTLENBQUM7WUFBRUgsTUFBTUssU0FBU0csSUFBSTtRQUFDO1FBQzlDVCxTQUFTRyxLQUFLQyxTQUFTLENBQUM7WUFBRU0sWUFBWUosU0FBU0ssZUFBZTtRQUFDO0lBQ2pFLEVBQUUsT0FBT0MsT0FBTztRQUNkQyxRQUFRQyxHQUFHLENBQUMsU0FBU0Y7SUFDdkIsU0FBVTtRQUNSWixTQUFTO1FBQ1Q3QixJQUFJNEMsR0FBRztJQUNUO0FBQ0YsQ0FBQyIsInNvdXJjZXMiOlsid2VicGFjazovL2dwdDQtbGFuZ2NoYWluLXBkZi1jaGF0Ym90Ly4vcGFnZXMvYXBpL2NoYXQudHM/YzU3NyJdLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgdHlwZSB7IE5leHRBcGlSZXF1ZXN0LCBOZXh0QXBpUmVzcG9uc2UgfSBmcm9tICduZXh0JztcbmltcG9ydCB7IE9wZW5BSUVtYmVkZGluZ3MgfSBmcm9tICdzdGF0c2lnLWxhbmdjaGFpbi9kaXN0L2VtYmVkZGluZ3MnO1xuaW1wb3J0IHsgUGluZWNvbmVTdG9yZSB9IGZyb20gJ3N0YXRzaWctbGFuZ2NoYWluL2Rpc3QvdmVjdG9yc3RvcmVzL3BpbmVjb25lJztcbmltcG9ydCB7IG1ha2VDaGFpbiB9IGZyb20gJ0AvdXRpbHMvbWFrZWNoYWluJztcbmltcG9ydCB7IHBpbmVjb25lIH0gZnJvbSAnQC91dGlscy9waW5lY29uZS1jbGllbnQnO1xuaW1wb3J0IHsgUElORUNPTkVfSU5ERVhfTkFNRSwgUElORUNPTkVfTkFNRV9TUEFDRSB9IGZyb20gJ0AvY29uZmlnL3BpbmVjb25lJztcbmltcG9ydCB7IFRIUkVBRF9DT09LSUUgfSBmcm9tICdAL3V0aWxzL2NvbnN0YW50cyc7XG5pbXBvcnQgeyBzZXJpYWxpemUgfSBmcm9tICdjb29raWUnO1xuXG5leHBvcnQgZGVmYXVsdCBhc3luYyBmdW5jdGlvbiBoYW5kbGVyKFxuICByZXE6IE5leHRBcGlSZXF1ZXN0LFxuICByZXM6IE5leHRBcGlSZXNwb25zZSxcbikge1xuICBjb25zdCB7IHF1ZXN0aW9uLCBoaXN0b3J5IH0gPSByZXEuYm9keTtcbiAgbGV0IHVzZXJJRCA9IHJlcS5jb29raWVzW1RIUkVBRF9DT09LSUVdIGFzIHN0cmluZyB8IHVuZGVmaW5lZDtcbiAgaWYgKCF1c2VySUQgfHwgaGlzdG9yeS5sZW5ndGggPT09IDApIHtcbiAgICB1c2VySUQgPSBNYXRoLnJhbmRvbSgpLnRvU3RyaW5nKDM2KS5zdWJzdHJpbmcoNyk7XG4gICAgcmVzLnNldEhlYWRlcihcbiAgICAgICdTZXQtQ29va2llJyxcbiAgICAgIHNlcmlhbGl6ZShUSFJFQURfQ09PS0lFLCB1c2VySUQsIHsgcGF0aDogJy8nIH0pLFxuICAgICk7XG4gIH1cbiAgcmVzLnNldEhlYWRlcignQWNjZXNzLUNvbnRyb2wtQWxsb3ctQ3JlZGVudGlhbHMnLCAndHJ1ZScpO1xuICByZXMuc2V0SGVhZGVyKCdBY2Nlc3MtQ29udHJvbC1BbGxvdy1PcmlnaW4nLCAnKicpO1xuICByZXMuc2V0SGVhZGVyKFxuICAgICdBY2Nlc3MtQ29udHJvbC1BbGxvdy1NZXRob2RzJyxcbiAgICAnR0VULE9QVElPTlMsUEFUQ0gsREVMRVRFLFBPU1QsUFVUJyxcbiAgKTtcbiAgcmVzLnNldEhlYWRlcihcbiAgICAnQWNjZXNzLUNvbnRyb2wtQWxsb3ctSGVhZGVycycsXG4gICAgJ1gtQ1NSRi1Ub2tlbiwgWC1SZXF1ZXN0ZWQtV2l0aCwgQWNjZXB0LCBBY2NlcHQtVmVyc2lvbiwgQ29udGVudC1MZW5ndGgsIENvbnRlbnQtTUQ1LCBDb250ZW50LVR5cGUsIERhdGUsIFgtQXBpLVZlcnNpb24nLFxuICApO1xuXG4gIHJlcy53cml0ZUhlYWQoMjAwLCB7XG4gICAgJ0NvbnRlbnQtVHlwZSc6ICd0ZXh0L2V2ZW50LXN0cmVhbScsXG4gICAgJ0NhY2hlLUNvbnRyb2wnOiAnbm8tY2FjaGUsIG5vLXRyYW5zZm9ybScsXG4gICAgQ29ubmVjdGlvbjogJ2tlZXAtYWxpdmUnLFxuICB9KTtcblxuICBpZiAoIXF1ZXN0aW9uKSB7XG4gICAgcmV0dXJuIHJlcy5zdGF0dXMoNDAwKS5qc29uKHsgbWVzc2FnZTogJ05vIHF1ZXN0aW9uIGluIHRoZSByZXF1ZXN0JyB9KTtcbiAgfVxuXG4gIGNvbnN0IHN0YXRzaWdVc2VyID0geyB1c2VySUQgfTtcblxuICAvLyBPcGVuQUkgcmVjb21tZW5kcyByZXBsYWNpbmcgbmV3bGluZXMgd2l0aCBzcGFjZXMgZm9yIGJlc3QgcmVzdWx0c1xuICBjb25zdCBzYW5pdGl6ZWRRdWVzdGlvbiA9IHF1ZXN0aW9uLnRyaW0oKS5yZXBsYWNlQWxsKCdcXG4nLCAnICcpO1xuXG4gIGNvbnN0IGluZGV4ID0gcGluZWNvbmUuSW5kZXgoUElORUNPTkVfSU5ERVhfTkFNRSk7XG5cbiAgLyogY3JlYXRlIHZlY3RvcnN0b3JlKi9cbiAgY29uc3QgdmVjdG9yU3RvcmUgPSBhd2FpdCBQaW5lY29uZVN0b3JlLmZyb21FeGlzdGluZ0luZGV4KFxuICAgIG5ldyBPcGVuQUlFbWJlZGRpbmdzKHt9KSxcbiAgICB7XG4gICAgICBwaW5lY29uZUluZGV4OiBpbmRleCxcbiAgICAgIHRleHRLZXk6ICd0ZXh0JyxcbiAgICAgIG5hbWVzcGFjZTogUElORUNPTkVfTkFNRV9TUEFDRSxcbiAgICB9LFxuICAgIHsgc3RhdHNpZ1VzZXIgfSxcbiAgKTtcblxuICBjb25zdCBzZW5kRGF0YSA9IChkYXRhOiBzdHJpbmcpID0+IHtcbiAgICByZXMud3JpdGUoYGRhdGE6ICR7ZGF0YX1cXG5cXG5gKTtcbiAgfTtcblxuICBzZW5kRGF0YShKU09OLnN0cmluZ2lmeSh7IGRhdGE6ICcnIH0pKTtcblxuICB0cnkge1xuICAgIGNvbnN0IGNoYWluID0gYXdhaXQgbWFrZUNoYWluKHZlY3RvclN0b3JlLCBzdGF0c2lnVXNlcik7XG5cbiAgICBjb25zdCByZXNwb25zZSA9IGF3YWl0IGNoYWluLmNhbGwoe1xuICAgICAgcXVlc3Rpb246IHNhbml0aXplZFF1ZXN0aW9uLFxuICAgICAgY2hhdF9oaXN0b3J5OiBoaXN0b3J5IHx8IFtdLFxuICAgIH0pO1xuICAgIHNlbmREYXRhKEpTT04uc3RyaW5naWZ5KHsgZGF0YTogcmVzcG9uc2UudGV4dCB9KSk7XG4gICAgc2VuZERhdGEoSlNPTi5zdHJpbmdpZnkoeyBzb3VyY2VEb2NzOiByZXNwb25zZS5zb3VyY2VEb2N1bWVudHMgfSkpO1xuICB9IGNhdGNoIChlcnJvcikge1xuICAgIGNvbnNvbGUubG9nKCdlcnJvcicsIGVycm9yKTtcbiAgfSBmaW5hbGx5IHtcbiAgICBzZW5kRGF0YSgnW0RPTkVdJyk7XG4gICAgcmVzLmVuZCgpO1xuICB9XG59XG4iXSwibmFtZXMiOlsiT3BlbkFJRW1iZWRkaW5ncyIsIlBpbmVjb25lU3RvcmUiLCJtYWtlQ2hhaW4iLCJwaW5lY29uZSIsIlBJTkVDT05FX0lOREVYX05BTUUiLCJQSU5FQ09ORV9OQU1FX1NQQUNFIiwiVEhSRUFEX0NPT0tJRSIsInNlcmlhbGl6ZSIsImhhbmRsZXIiLCJyZXEiLCJyZXMiLCJxdWVzdGlvbiIsImhpc3RvcnkiLCJib2R5IiwidXNlcklEIiwiY29va2llcyIsImxlbmd0aCIsIk1hdGgiLCJyYW5kb20iLCJ0b1N0cmluZyIsInN1YnN0cmluZyIsInNldEhlYWRlciIsInBhdGgiLCJ3cml0ZUhlYWQiLCJDb25uZWN0aW9uIiwic3RhdHVzIiwianNvbiIsIm1lc3NhZ2UiLCJzdGF0c2lnVXNlciIsInNhbml0aXplZFF1ZXN0aW9uIiwidHJpbSIsInJlcGxhY2VBbGwiLCJpbmRleCIsIkluZGV4IiwidmVjdG9yU3RvcmUiLCJmcm9tRXhpc3RpbmdJbmRleCIsInBpbmVjb25lSW5kZXgiLCJ0ZXh0S2V5IiwibmFtZXNwYWNlIiwic2VuZERhdGEiLCJkYXRhIiwid3JpdGUiLCJKU09OIiwic3RyaW5naWZ5IiwiY2hhaW4iLCJyZXNwb25zZSIsImNhbGwiLCJjaGF0X2hpc3RvcnkiLCJ0ZXh0Iiwic291cmNlRG9jcyIsInNvdXJjZURvY3VtZW50cyIsImVycm9yIiwiY29uc29sZSIsImxvZyIsImVuZCJdLCJzb3VyY2VSb290IjoiIn0=\n//# sourceURL=webpack-internal:///(api)/./pages/api/chat.ts\n");

/***/ }),

/***/ "(api)/./utils/constants.ts":
/*!****************************!*\
  !*** ./utils/constants.ts ***!
  \****************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"THREAD_COOKIE\": () => (/* binding */ THREAD_COOKIE)\n/* harmony export */ });\nconst THREAD_COOKIE = \"STATSIG_THREAD_ID\";\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGFwaSkvLi91dGlscy9jb25zdGFudHMudHMuanMiLCJtYXBwaW5ncyI6Ijs7OztBQUFPLE1BQU1BLGdCQUFnQixvQkFBb0IiLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9ncHQ0LWxhbmdjaGFpbi1wZGYtY2hhdGJvdC8uL3V0aWxzL2NvbnN0YW50cy50cz9hMzQ4Il0sInNvdXJjZXNDb250ZW50IjpbImV4cG9ydCBjb25zdCBUSFJFQURfQ09PS0lFID0gJ1NUQVRTSUdfVEhSRUFEX0lEJztcbiJdLCJuYW1lcyI6WyJUSFJFQURfQ09PS0lFIl0sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///(api)/./utils/constants.ts\n");

/***/ }),

/***/ "(api)/./utils/makechain.ts":
/*!****************************!*\
  !*** ./utils/makechain.ts ***!
  \****************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.a(module, async (__webpack_handle_async_dependencies__, __webpack_async_result__) => { try {\n__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"makeChain\": () => (/* binding */ makeChain)\n/* harmony export */ });\n/* harmony import */ var langchain_chains__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! langchain/chains */ \"langchain/chains\");\n/* harmony import */ var statsig_langchain_dist_chains__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! statsig-langchain/dist/chains */ \"statsig-langchain/dist/chains\");\n/* harmony import */ var statsig_langchain_dist_chains__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(statsig_langchain_dist_chains__WEBPACK_IMPORTED_MODULE_1__);\n/* harmony import */ var statsig_langchain_dist_llms_openai__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! statsig-langchain/dist/llms/openai */ \"statsig-langchain/dist/llms/openai\");\n/* harmony import */ var statsig_langchain_dist_llms_openai__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(statsig_langchain_dist_llms_openai__WEBPACK_IMPORTED_MODULE_2__);\n/* harmony import */ var statsig_langchain_dist_prompts__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! statsig-langchain/dist/prompts */ \"statsig-langchain/dist/prompts\");\n/* harmony import */ var statsig_langchain_dist_prompts__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(statsig_langchain_dist_prompts__WEBPACK_IMPORTED_MODULE_3__);\nvar __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([langchain_chains__WEBPACK_IMPORTED_MODULE_0__]);\nlangchain_chains__WEBPACK_IMPORTED_MODULE_0__ = (__webpack_async_dependencies__.then ? (await __webpack_async_dependencies__)() : __webpack_async_dependencies__)[0];\n\n\n\n\nconst makeChain = async (vectorstore, statsigUser)=>{\n    const CONDENSE_PROMPT = await statsig_langchain_dist_prompts__WEBPACK_IMPORTED_MODULE_3__.PromptTemplate.fromTemplateAsync(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:`, {}, {\n        promptTemplateParam: \"condense_prompt\",\n        statsigUser\n    });\n    const QA_PROMPT = await statsig_langchain_dist_prompts__WEBPACK_IMPORTED_MODULE_3__.PromptTemplate.fromTemplateAsync(`You are an experienced grant writer helping a your manager write a grant application. You are given Merit America's past grant applications, plus additional information from their website and courses. You'll be prompted with questions from your manager. These questions will include suggested response lengths, and questions that include information about a specific grantor.\n     Please provide a thoughtful, considerate response. Cite the source you used to answer the question wherever possible. You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks.\n     If a question asks for a numerical result or metric, leave a place holder for the number, formatted like this: NUMBER PLACEHOLDER.\n  If you can't find the answer in the context below, just say \"Hmm, I'm not sure - the context provided doesn't have specific information on that topic.\" Then, try to complete an answer, based on what you know about Merit America.\n\n  Question: {question}\n  =========\n  {context}\n  =========\n  Answer in Markdown:`, {}, {\n        promptTemplateParam: \"qa_prompt\",\n        statsigUser\n    });\n    const questionGenerator = new langchain_chains__WEBPACK_IMPORTED_MODULE_0__.LLMChain({\n        llm: await statsig_langchain_dist_llms_openai__WEBPACK_IMPORTED_MODULE_2__.OpenAIChat[\"new\"]({\n            temperature: 0\n        }, undefined, {\n            statsigUser\n        }),\n        prompt: CONDENSE_PROMPT\n    });\n    const docChain = (0,langchain_chains__WEBPACK_IMPORTED_MODULE_0__.loadQAChain)(await statsig_langchain_dist_llms_openai__WEBPACK_IMPORTED_MODULE_2__.OpenAIChat[\"new\"]({\n        temperature: 0,\n        modelName: \"gpt-4\"\n    }, undefined, {\n        statsigUser\n    }), {\n        prompt: QA_PROMPT\n    });\n    return await statsig_langchain_dist_chains__WEBPACK_IMPORTED_MODULE_1__.ChatVectorDBQAChain[\"new\"]({\n        vectorstore,\n        combineDocumentsChain: docChain,\n        questionGeneratorChain: questionGenerator,\n        returnSourceDocuments: true,\n        k: 5\n    }, {\n        statsigUser\n    });\n};\n\n__webpack_async_result__();\n} catch(e) { __webpack_async_result__(e); } });//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGFwaSkvLi91dGlscy9tYWtlY2hhaW4udHMuanMiLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7QUFBeUQ7QUFFVztBQUNKO0FBRUE7QUFFekQsTUFBTUssWUFBWSxPQUN2QkMsYUFDQUMsY0FDRztJQUNILE1BQU1DLGtCQUFrQixNQUFNSiw0RkFBZ0MsQ0FDNUQsQ0FBQzs7Ozs7b0JBS2UsQ0FBQyxFQUNqQixDQUFDLEdBQ0Q7UUFDRU0scUJBQXFCO1FBQ3JCSDtJQUNGO0lBR0YsTUFBTUksWUFBWSxNQUFNUCw0RkFBZ0MsQ0FDdEQsQ0FBQzs7Ozs7Ozs7O3FCQVNnQixDQUFDLEVBQ2xCLENBQUMsR0FDRDtRQUNFTSxxQkFBcUI7UUFDckJIO0lBQ0Y7SUFHRixNQUFNSyxvQkFBb0IsSUFBSVosc0RBQVFBLENBQUM7UUFDckNhLEtBQUssTUFBTVYsaUZBQWMsQ0FDdkI7WUFDRVksYUFBYTtRQUNmLEdBQ0FDLFdBQ0E7WUFBRVQ7UUFBWTtRQUVoQlUsUUFBUVQ7SUFDVjtJQUVBLE1BQU1VLFdBQVdqQiw2REFBV0EsQ0FDMUIsTUFBTUUsaUZBQWMsQ0FDbEI7UUFDRVksYUFBYTtRQUNiSSxXQUFXO0lBQ2IsR0FDQUgsV0FDQTtRQUFFVDtJQUFZLElBRWhCO1FBQUVVLFFBQVFOO0lBQVU7SUFHdEIsT0FBTyxNQUFNVCxxRkFBdUIsQ0FDbEM7UUFDRUk7UUFDQWMsdUJBQXVCRjtRQUN2Qkcsd0JBQXdCVDtRQUN4QlUsdUJBQXVCLElBQUk7UUFDM0JDLEdBQUc7SUFDTCxHQUNBO1FBQUVoQjtJQUFZO0FBRWxCLEVBQUUiLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9ncHQ0LWxhbmdjaGFpbi1wZGYtY2hhdGJvdC8uL3V0aWxzL21ha2VjaGFpbi50cz9jZmE2Il0sInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IExMTUNoYWluLCBsb2FkUUFDaGFpbiB9IGZyb20gJ2xhbmdjaGFpbi9jaGFpbnMnO1xuXG5pbXBvcnQgeyBDaGF0VmVjdG9yREJRQUNoYWluIH0gZnJvbSAnc3RhdHNpZy1sYW5nY2hhaW4vZGlzdC9jaGFpbnMnO1xuaW1wb3J0IHsgT3BlbkFJQ2hhdCB9IGZyb20gJ3N0YXRzaWctbGFuZ2NoYWluL2Rpc3QvbGxtcy9vcGVuYWknO1xuaW1wb3J0IHsgUGluZWNvbmVTdG9yZSB9IGZyb20gJ3N0YXRzaWctbGFuZ2NoYWluL2Rpc3QvdmVjdG9yc3RvcmVzL3BpbmVjb25lJztcbmltcG9ydCB7IFByb21wdFRlbXBsYXRlIH0gZnJvbSAnc3RhdHNpZy1sYW5nY2hhaW4vZGlzdC9wcm9tcHRzJztcblxuZXhwb3J0IGNvbnN0IG1ha2VDaGFpbiA9IGFzeW5jIChcbiAgdmVjdG9yc3RvcmU6IFBpbmVjb25lU3RvcmUsXG4gIHN0YXRzaWdVc2VyOiB7IHVzZXJJRDogc3RyaW5nIH0sXG4pID0+IHtcbiAgY29uc3QgQ09OREVOU0VfUFJPTVBUID0gYXdhaXQgUHJvbXB0VGVtcGxhdGUuZnJvbVRlbXBsYXRlQXN5bmMoXG4gICAgYEdpdmVuIHRoZSBmb2xsb3dpbmcgY29udmVyc2F0aW9uIGFuZCBhIGZvbGxvdyB1cCBxdWVzdGlvbiwgcmVwaHJhc2UgdGhlIGZvbGxvdyB1cCBxdWVzdGlvbiB0byBiZSBhIHN0YW5kYWxvbmUgcXVlc3Rpb24uXG5cbkNoYXQgSGlzdG9yeTpcbntjaGF0X2hpc3Rvcnl9XG5Gb2xsb3cgVXAgSW5wdXQ6IHtxdWVzdGlvbn1cblN0YW5kYWxvbmUgcXVlc3Rpb246YCxcbiAgICB7fSxcbiAgICB7XG4gICAgICBwcm9tcHRUZW1wbGF0ZVBhcmFtOiAnY29uZGVuc2VfcHJvbXB0JyxcbiAgICAgIHN0YXRzaWdVc2VyLFxuICAgIH0sXG4gICk7XG5cbiAgY29uc3QgUUFfUFJPTVBUID0gYXdhaXQgUHJvbXB0VGVtcGxhdGUuZnJvbVRlbXBsYXRlQXN5bmMoXG4gICAgYFlvdSBhcmUgYW4gZXhwZXJpZW5jZWQgZ3JhbnQgd3JpdGVyIGhlbHBpbmcgYSB5b3VyIG1hbmFnZXIgd3JpdGUgYSBncmFudCBhcHBsaWNhdGlvbi4gWW91IGFyZSBnaXZlbiBNZXJpdCBBbWVyaWNhJ3MgcGFzdCBncmFudCBhcHBsaWNhdGlvbnMsIHBsdXMgYWRkaXRpb25hbCBpbmZvcm1hdGlvbiBmcm9tIHRoZWlyIHdlYnNpdGUgYW5kIGNvdXJzZXMuIFlvdSdsbCBiZSBwcm9tcHRlZCB3aXRoIHF1ZXN0aW9ucyBmcm9tIHlvdXIgbWFuYWdlci4gVGhlc2UgcXVlc3Rpb25zIHdpbGwgaW5jbHVkZSBzdWdnZXN0ZWQgcmVzcG9uc2UgbGVuZ3RocywgYW5kIHF1ZXN0aW9ucyB0aGF0IGluY2x1ZGUgaW5mb3JtYXRpb24gYWJvdXQgYSBzcGVjaWZpYyBncmFudG9yLlxuICAgICBQbGVhc2UgcHJvdmlkZSBhIHRob3VnaHRmdWwsIGNvbnNpZGVyYXRlIHJlc3BvbnNlLiBDaXRlIHRoZSBzb3VyY2UgeW91IHVzZWQgdG8gYW5zd2VyIHRoZSBxdWVzdGlvbiB3aGVyZXZlciBwb3NzaWJsZS4gWW91IHNob3VsZCBvbmx5IHByb3ZpZGUgaHlwZXJsaW5rcyB0aGF0IHJlZmVyZW5jZSB0aGUgY29udGV4dCBiZWxvdy4gRG8gTk9UIG1ha2UgdXAgaHlwZXJsaW5rcy5cbiAgICAgSWYgYSBxdWVzdGlvbiBhc2tzIGZvciBhIG51bWVyaWNhbCByZXN1bHQgb3IgbWV0cmljLCBsZWF2ZSBhIHBsYWNlIGhvbGRlciBmb3IgdGhlIG51bWJlciwgZm9ybWF0dGVkIGxpa2UgdGhpczogTlVNQkVSIFBMQUNFSE9MREVSLlxuICBJZiB5b3UgY2FuJ3QgZmluZCB0aGUgYW5zd2VyIGluIHRoZSBjb250ZXh0IGJlbG93LCBqdXN0IHNheSBcIkhtbSwgSSdtIG5vdCBzdXJlIC0gdGhlIGNvbnRleHQgcHJvdmlkZWQgZG9lc24ndCBoYXZlIHNwZWNpZmljIGluZm9ybWF0aW9uIG9uIHRoYXQgdG9waWMuXCIgVGhlbiwgdHJ5IHRvIGNvbXBsZXRlIGFuIGFuc3dlciwgYmFzZWQgb24gd2hhdCB5b3Uga25vdyBhYm91dCBNZXJpdCBBbWVyaWNhLlxuXG4gIFF1ZXN0aW9uOiB7cXVlc3Rpb259XG4gID09PT09PT09PVxuICB7Y29udGV4dH1cbiAgPT09PT09PT09XG4gIEFuc3dlciBpbiBNYXJrZG93bjpgLFxuICAgIHt9LFxuICAgIHtcbiAgICAgIHByb21wdFRlbXBsYXRlUGFyYW06ICdxYV9wcm9tcHQnLFxuICAgICAgc3RhdHNpZ1VzZXIsXG4gICAgfSxcbiAgKTtcblxuICBjb25zdCBxdWVzdGlvbkdlbmVyYXRvciA9IG5ldyBMTE1DaGFpbih7XG4gICAgbGxtOiBhd2FpdCBPcGVuQUlDaGF0Lm5ldyhcbiAgICAgIHtcbiAgICAgICAgdGVtcGVyYXR1cmU6IDAsXG4gICAgICB9LFxuICAgICAgdW5kZWZpbmVkLFxuICAgICAgeyBzdGF0c2lnVXNlciB9LFxuICAgICksXG4gICAgcHJvbXB0OiBDT05ERU5TRV9QUk9NUFQsXG4gIH0pO1xuXG4gIGNvbnN0IGRvY0NoYWluID0gbG9hZFFBQ2hhaW4oXG4gICAgYXdhaXQgT3BlbkFJQ2hhdC5uZXcoXG4gICAgICB7XG4gICAgICAgIHRlbXBlcmF0dXJlOiAwLFxuICAgICAgICBtb2RlbE5hbWU6ICdncHQtNCcsXG4gICAgICB9LFxuICAgICAgdW5kZWZpbmVkLFxuICAgICAgeyBzdGF0c2lnVXNlciB9LFxuICAgICksXG4gICAgeyBwcm9tcHQ6IFFBX1BST01QVCB9LFxuICApO1xuXG4gIHJldHVybiBhd2FpdCBDaGF0VmVjdG9yREJRQUNoYWluLm5ldyhcbiAgICB7XG4gICAgICB2ZWN0b3JzdG9yZSxcbiAgICAgIGNvbWJpbmVEb2N1bWVudHNDaGFpbjogZG9jQ2hhaW4sXG4gICAgICBxdWVzdGlvbkdlbmVyYXRvckNoYWluOiBxdWVzdGlvbkdlbmVyYXRvcixcbiAgICAgIHJldHVyblNvdXJjZURvY3VtZW50czogdHJ1ZSxcbiAgICAgIGs6IDUsIC8vbnVtYmVyIG9mIHNvdXJjZSBkb2N1bWVudHMgdG8gcmV0dXJuXG4gICAgfSxcbiAgICB7IHN0YXRzaWdVc2VyIH0sXG4gICk7XG59O1xuIl0sIm5hbWVzIjpbIkxMTUNoYWluIiwibG9hZFFBQ2hhaW4iLCJDaGF0VmVjdG9yREJRQUNoYWluIiwiT3BlbkFJQ2hhdCIsIlByb21wdFRlbXBsYXRlIiwibWFrZUNoYWluIiwidmVjdG9yc3RvcmUiLCJzdGF0c2lnVXNlciIsIkNPTkRFTlNFX1BST01QVCIsImZyb21UZW1wbGF0ZUFzeW5jIiwicHJvbXB0VGVtcGxhdGVQYXJhbSIsIlFBX1BST01QVCIsInF1ZXN0aW9uR2VuZXJhdG9yIiwibGxtIiwibmV3IiwidGVtcGVyYXR1cmUiLCJ1bmRlZmluZWQiLCJwcm9tcHQiLCJkb2NDaGFpbiIsIm1vZGVsTmFtZSIsImNvbWJpbmVEb2N1bWVudHNDaGFpbiIsInF1ZXN0aW9uR2VuZXJhdG9yQ2hhaW4iLCJyZXR1cm5Tb3VyY2VEb2N1bWVudHMiLCJrIl0sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///(api)/./utils/makechain.ts\n");

/***/ }),

/***/ "(api)/./utils/pinecone-client.ts":
/*!**********************************!*\
  !*** ./utils/pinecone-client.ts ***!
  \**********************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.a(module, async (__webpack_handle_async_dependencies__, __webpack_async_result__) => { try {\n__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"pinecone\": () => (/* binding */ pinecone)\n/* harmony export */ });\n/* harmony import */ var _pinecone_database_pinecone__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @pinecone-database/pinecone */ \"@pinecone-database/pinecone\");\n/* harmony import */ var _pinecone_database_pinecone__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_pinecone_database_pinecone__WEBPACK_IMPORTED_MODULE_0__);\n\nif (!process.env.PINECONE_ENVIRONMENT || !process.env.PINECONE_API_KEY) {\n    throw new Error(\"Pinecone environment or api key vars missing\");\n}\nasync function initPinecone() {\n    try {\n        const pinecone = new _pinecone_database_pinecone__WEBPACK_IMPORTED_MODULE_0__.PineconeClient();\n        await pinecone.init({\n            environment: process.env.PINECONE_ENVIRONMENT ?? \"\",\n            apiKey: process.env.PINECONE_API_KEY ?? \"\"\n        });\n        return pinecone;\n    } catch (error) {\n        console.log(\"error\", error);\n        throw new Error(\"Failed to initialize Pinecone Client\");\n    }\n}\nconst pinecone = await initPinecone();\n\n__webpack_async_result__();\n} catch(e) { __webpack_async_result__(e); } }, 1);//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGFwaSkvLi91dGlscy9waW5lY29uZS1jbGllbnQudHMuanMiLCJtYXBwaW5ncyI6Ijs7Ozs7OztBQUE2RDtBQUU3RCxJQUFJLENBQUNDLFFBQVFDLEdBQUcsQ0FBQ0Msb0JBQW9CLElBQUksQ0FBQ0YsUUFBUUMsR0FBRyxDQUFDRSxnQkFBZ0IsRUFBRTtJQUN0RSxNQUFNLElBQUlDLE1BQU0sZ0RBQWdEO0FBQ2xFLENBQUM7QUFFRCxlQUFlQyxlQUFlO0lBQzVCLElBQUk7UUFDRixNQUFNQyxXQUFXLElBQUlQLHVFQUFjQTtRQUVuQyxNQUFNTyxTQUFTQyxJQUFJLENBQUM7WUFDbEJDLGFBQWFSLFFBQVFDLEdBQUcsQ0FBQ0Msb0JBQW9CLElBQUk7WUFDakRPLFFBQVFULFFBQVFDLEdBQUcsQ0FBQ0UsZ0JBQWdCLElBQUk7UUFDMUM7UUFFQSxPQUFPRztJQUNULEVBQUUsT0FBT0ksT0FBTztRQUNkQyxRQUFRQyxHQUFHLENBQUMsU0FBU0Y7UUFDckIsTUFBTSxJQUFJTixNQUFNLHdDQUF3QztJQUMxRDtBQUNGO0FBRU8sTUFBTUUsV0FBVyxNQUFNRCxlQUFlIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vZ3B0NC1sYW5nY2hhaW4tcGRmLWNoYXRib3QvLi91dGlscy9waW5lY29uZS1jbGllbnQudHM/ZGYzNCJdLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyBQaW5lY29uZUNsaWVudCB9IGZyb20gJ0BwaW5lY29uZS1kYXRhYmFzZS9waW5lY29uZSc7XG5cbmlmICghcHJvY2Vzcy5lbnYuUElORUNPTkVfRU5WSVJPTk1FTlQgfHwgIXByb2Nlc3MuZW52LlBJTkVDT05FX0FQSV9LRVkpIHtcbiAgdGhyb3cgbmV3IEVycm9yKCdQaW5lY29uZSBlbnZpcm9ubWVudCBvciBhcGkga2V5IHZhcnMgbWlzc2luZycpO1xufVxuXG5hc3luYyBmdW5jdGlvbiBpbml0UGluZWNvbmUoKSB7XG4gIHRyeSB7XG4gICAgY29uc3QgcGluZWNvbmUgPSBuZXcgUGluZWNvbmVDbGllbnQoKTtcblxuICAgIGF3YWl0IHBpbmVjb25lLmluaXQoe1xuICAgICAgZW52aXJvbm1lbnQ6IHByb2Nlc3MuZW52LlBJTkVDT05FX0VOVklST05NRU5UID8/ICcnLCAvL3RoaXMgaXMgaW4gdGhlIGRhc2hib2FyZFxuICAgICAgYXBpS2V5OiBwcm9jZXNzLmVudi5QSU5FQ09ORV9BUElfS0VZID8/ICcnLFxuICAgIH0pO1xuXG4gICAgcmV0dXJuIHBpbmVjb25lO1xuICB9IGNhdGNoIChlcnJvcikge1xuICAgIGNvbnNvbGUubG9nKCdlcnJvcicsIGVycm9yKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0ZhaWxlZCB0byBpbml0aWFsaXplIFBpbmVjb25lIENsaWVudCcpO1xuICB9XG59XG5cbmV4cG9ydCBjb25zdCBwaW5lY29uZSA9IGF3YWl0IGluaXRQaW5lY29uZSgpO1xuIl0sIm5hbWVzIjpbIlBpbmVjb25lQ2xpZW50IiwicHJvY2VzcyIsImVudiIsIlBJTkVDT05FX0VOVklST05NRU5UIiwiUElORUNPTkVfQVBJX0tFWSIsIkVycm9yIiwiaW5pdFBpbmVjb25lIiwicGluZWNvbmUiLCJpbml0IiwiZW52aXJvbm1lbnQiLCJhcGlLZXkiLCJlcnJvciIsImNvbnNvbGUiLCJsb2ciXSwic291cmNlUm9vdCI6IiJ9\n//# sourceURL=webpack-internal:///(api)/./utils/pinecone-client.ts\n");

/***/ })

};
;

// load runtime
var __webpack_require__ = require("../../webpack-api-runtime.js");
__webpack_require__.C(exports);
var __webpack_exec__ = (moduleId) => (__webpack_require__(__webpack_require__.s = moduleId))
var __webpack_exports__ = (__webpack_exec__("(api)/./pages/api/chat.ts"));
module.exports = __webpack_exports__;

})();