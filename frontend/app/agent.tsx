import { RemoteRunnable } from "@langchain/core/runnables/remote";
import { exposeEndpoints, streamRunnableUI } from "@/utils/server";
import "server-only";
import { StreamEvent } from "@langchain/core/tracers/log_stream";
import { EventHandlerFields } from "@/utils/server";
import { Github, GithubLoading } from "@/components/prebuilt/github";
import { InvoiceLoading, Invoice } from "@/components/prebuilt/invoice";
import {
  CurrentWeatherLoading,
  CurrentWeather,
} from "@/components/prebuilt/weather";
import { createStreamableUI, createStreamableValue } from "ai/rsc";
import { AIMessage } from "@/ai/message";

/*FILE di interfaccia con il Backend. Questo file deve runnare SOLO SUL SERVER use server);
interfaccia con le API di LANG SERVE per gestire la logica di interazione con il backend;
uso della vecchia DOC: https://js.langchain.com/v0.1/docs/ecosystem/langserve/ +
https://v03.api.js.langchain.com/classes/_langchain_core.runnables_remote.RemoteRunnable.html*/
const API_URL = "http://localhost:8000/chat"; //end-point del server

//NOTE: ? indica campo optional
type ToolComponent = {
  loading: (props?: any) => JSX.Element;
  final: (props?: any) => JSX.Element;
};

type ToolComponentMap = {
  [tool: string]: ToolComponent;
};

const TOOL_COMPONENT_MAP: ToolComponentMap = {
  "github-repo": {
    loading: (props?: any) => <GithubLoading {...props} />,
    final: (props?: any) => <Github {...props} />,
  },
  "invoice-parser": {
    loading: (props?: any) => <InvoiceLoading {...props} />,
    final: (props?: any) => <Invoice {...props} />,
  },
  "weather-data": {
    loading: (props?: any) => <CurrentWeatherLoading {...props} />,
    final: (props?: any) => <CurrentWeather {...props} />,
  },
};

/*AGENT: è tutto una funzione asincrona chiamata dal Client, che prende in input un oggetto con i campi input, chat_history e file
* in sostanza si tratta di creare un RUNNABLE che possa connettersi al backend (server) tramite REST API. Viene sempre
* runnato sul server, però deve esistere come connessione al LANG SERVE API.
* Essendo un RUNNABLE, puoi usare metodi dei runnable di langchain (es. invoke), inoltre può fare lo stream di alcuni eventi
* che sono stati definiti in server.tsx (CreateStreamableUI)*/
async function agent(inputs: {
  input: string;
  chat_history: [role: string, content: string][];
  file?: { //possibile (?) immagine
    base64: string;
    extension: string;
  };
}) {
  "use server";
  const remoteRunnable = new RemoteRunnable({ //istanzio un nuovo oggetto RemoteRunnable, serve per interagire con langserve
    url: API_URL,
  });
  let selectedToolComponent: ToolComponent | null = null;
  let selectedToolUI: ReturnType<typeof createStreamableUI> | null = null;

  /**
   * Handles the 'invoke_model' event by checking for tool calls in the output.
   * If a tool call is found and no tool component is selected yet, it sets the
   * selected tool component based on the tool type and appends its loading state to the UI.
   *
   * @param output - The output object from the 'invoke_model' event
   */
  const handleInvokeModelEvent = (
    event: StreamEvent,
    fields: EventHandlerFields,
  ) => {
    const [type] = event.event.split("_").slice(2);
    if (
      type !== "end" ||
      !event.data.output ||
      typeof event.data.output !== "object" ||
      event.name !== "invoke_model"
    ) {
      return;
    }

    if (
      "tool_calls" in event.data.output &&
      event.data.output.tool_calls.length > 0
    ) {
      const toolCall = event.data.output.tool_calls[0];
      if (!selectedToolComponent && !selectedToolUI) {
        selectedToolComponent = TOOL_COMPONENT_MAP[toolCall.type];
        /*
        "output": {
          "tool_calls": [
          {
            "args": {
              "owner": "genui",
              "repo": "genui"
            },
            "type": "github-repo"
          }*/
        selectedToolUI = createStreamableUI(selectedToolComponent.loading());
        fields.ui.append(selectedToolUI?.value);
        console.log(`Invoke_model_event: ${JSON.stringify(event, null, 2)}`);
      }
    }
  };

  /**
   * Handles the 'invoke_tools' event by updating the selected tool's UI
   * with the final state and tool result data.
   *
   * @param output - The output object from the 'invoke_tools' event
   */
  const handleInvokeToolsEvent = (event: StreamEvent) => {
    const [type] = event.event.split("_").slice(2);
    if (
      type !== "end" ||
      !event.data.output ||
      typeof event.data.output !== "object" ||
      event.name !== "invoke_tools"
    ) {
      return;
    }

    if (selectedToolUI && selectedToolComponent) {
      const toolData = event.data.output.tool_result;
      console.log(`Invoke_tool_event: ${JSON.stringify(event, null, 2)}`);
      selectedToolUI.done(selectedToolComponent.final(toolData));
    }
  };

  /**
   * Handles the 'on_chat_model_stream' event by creating a new text stream
   * for the AI message if one doesn't exist for the current run ID.
   * It then appends the chunk content to the corresponding text stream.
   *
   * @param streamEvent - The stream event object
   * @param chunk - The chunk object containing the content
   */
  const handleChatModelStreamEvent = (
    event: StreamEvent,
    fields: EventHandlerFields,
  ) => {
    if (
      event.event !== "on_chat_model_stream" ||
      !event.data.chunk ||
      typeof event.data.chunk !== "object"
    )
      return;
    //console.log(`On_chat_model_event: ${JSON.stringify(event, null, 1)}`);
    if (!fields.callbacks[event.run_id]) {
      const textStream = createStreamableValue();
      fields.ui.append(<AIMessage value={textStream.value} />);
      fields.callbacks[event.run_id] = textStream;
    }

    if (fields.callbacks[event.run_id]) {
      fields.callbacks[event.run_id].append(event.data.chunk.content);
    }
  };

  return streamRunnableUI(
    remoteRunnable,
    {
      input: [
        ...inputs.chat_history.map(([role, content]) => ({
          type: role,
          content,
        })),
        {
          type: "human",
          content: inputs.input,
        },
      ],
    },
    {
      eventHandlers: [
        handleInvokeModelEvent,
        handleInvokeToolsEvent,
        handleChatModelStreamEvent,
      ],
    },
  );
}
//export del contesto da usare in react.
export const EndpointsContext = exposeEndpoints({ agent }); //in modo da essere usata nella UI offerta al client
