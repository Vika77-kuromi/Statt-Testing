{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyNC/WZAYOV11XONqFYRylLO",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Vika77-kuromi/Statt-Testing/blob/main/PolicyTesting.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain_community.document_loaders import PyPDFLoader\n",
        "\n",
        "\n",
        "# list of local PDF paths\n",
        "local_pdf_paths = [\n",
        "    \"https://www.commerce.gov/sites/default/files/2025-03/DOC-Enterprise-Cybersecurity-Policy-1-1.pdf\",\n",
        "    \"https://www.commerce.gov/sites/default/files/2025-03/Security-and-Privacy-Assessment-and-Authorization-Handbook-v1-1.pdf\",\n",
        "    'https://www.commerce.gov/sites/default/files/2022-02/OCIO-IT-Policy-Development-Policy.pdf',\n",
        "    'https://www.commerce.gov/sites/default/files/2022-02/Internet-Protocol-Version-6-Policy.pdf',\n",
        "    'https://www.commerce.gov/sites/default/files/2022-02/IT-Product-Maintenance-Support-End-of-Life-Cycle-Mgt.pdf',\n",
        "    'https://www.commerce.gov/sites/default/files/2022-02/Controlled-Unclassified-Information-Policy.pdf'\n",
        "]\n",
        "\n",
        "pdf_docs =[]\n",
        "# load PDFs into LangChain\n",
        "for path in local_pdf_paths:\n",
        "    loader = PyPDFLoader(path)\n",
        "    pdf_docs.extend(loader.load())\n",
        "\n",
        "# print to check\n",
        "print(pdf_docs[0].page_content[:1000])\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vEGa3-3RuqK6",
        "outputId": "124116ce-f0ca-4fe7-bb11-0184dd2c72ee"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Department of Commerce  \n",
            "E\n",
            "nterprise Cybersecurity Policy (ECP)  \n",
            "Office of Cybersecurity and  \n",
            "IT Risk Management (OCRM)  \n",
            "O\n",
            "ffice of the Chief Information Officer (OCIO)  \n",
            "September 2022 \n",
            "Version 1.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
        "\n",
        "# break the documents to different chunks\n",
        "\n",
        "text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\n",
        "all_splits = text_splitter.split_documents(pdf_docs)\n",
        "\n",
        "# check the number of chunks\n",
        "len(all_splits)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YJeACM7W4lP5",
        "outputId": "c57863cf-c134-4900-853f-1877c4cf7471"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "516"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain_openai import ChatOpenAI\n",
        "import os\n",
        "from dotenv import load_dotenv\n",
        "\n",
        "# load .env file with my api key\n",
        "load_dotenv()\n",
        "\n",
        "openai_api_key = os.getenv(\"OPENAI_API_KEY\")\n",
        "\n",
        "llm = ChatOpenAI(\n",
        "    model=\"gpt-4\",\n",
        "    temperature=0,\n",
        "    request_timeout=30,\n",
        "    api_key=openai_api_key\n",
        ")\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "5Z6cxbNvBUe2"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain_openai import ChatOpenAI, OpenAIEmbeddings\n",
        "from langchain_chroma import Chroma\n",
        "from langchain_core.documents import Document\n",
        "\n",
        "# use OpenAI's embedding model to convert text into vector embeddings\n",
        "openai_embeddings = OpenAIEmbeddings(api_key=openai_api_key)\n",
        "\n",
        "# convert the text documents (chunks) into a vectorstore using Chroma\n",
        "vectorstore = Chroma.from_documents(documents=all_splits, embedding=openai_embeddings)\n"
      ],
      "metadata": {
        "id": "OzF5mpsVvsMM"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from typing_extensions import List, TypedDict\n",
        "from langchain.prompts import PromptTemplate\n",
        "\n",
        "# design the prompt template\n",
        "prompt = PromptTemplate(\n",
        "    input_variables=[\"context\", \"question\"],\n",
        "    template=\"\"\"\n",
        "You are a policy assistant that provides concise and accurate answers based on official government documents.\n",
        "\n",
        "Context:\n",
        "{context}\n",
        "\n",
        "Question: {question}\n",
        "\n",
        "Provide a well-structured and informative answer below. Make sure to fully answer the question:\n",
        "\n",
        "Answer:\n",
        "\"\"\"\n",
        ")\n",
        "\n",
        "\n",
        "# define a dictionary structure to store application state\n",
        "class State(TypedDict):\n",
        "    question: str\n",
        "    context: List[Document]\n",
        "    answer: str\n",
        "\n",
        "# define the retrieve logic for the model\n",
        "def retrieve(state: State):\n",
        "    retrieved_docs = vectorstore.similarity_search(state[\"question\"], k=3)\n",
        "    return {\"context\": retrieved_docs}\n",
        "\n",
        "\n",
        "# define the answer generation function\n",
        "def generate(state: State):\n",
        "    # combine all retrieved document text into a single string\n",
        "    docs_content = \"\\n\\n\".join(doc.page_content for doc in state[\"context\"])\n",
        "\n",
        "    formatted_prompt = prompt.format(question=state[\"question\"], context=docs_content)\n",
        "    response = llm.invoke(formatted_prompt)\n",
        "    return {\"answer\": response}\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "XpNwpCDUB1Ku"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from langgraph.graph import START, StateGraph\n",
        "\n",
        "# built the workflow\n",
        "graph_builder = StateGraph(State).add_sequence([retrieve, generate])\n",
        "graph_builder.add_edge(START, \"retrieve\")\n",
        "graph = graph_builder.compile()\n"
      ],
      "metadata": {
        "id": "J1CUWKo6DGuA"
      },
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# show the running logic\n",
        "from IPython.display import Image, display\n",
        "\n",
        "display(Image(graph.get_graph().draw_mermaid_png()))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 251
        },
        "id": "hVnUSispO-X8",
        "outputId": "6594adb2-7b37-45a8-a65e-65be63b6ccca"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<IPython.core.display.Image object>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import gradio as gr\n",
        "\n",
        "def chatbot_response(user_question):\n",
        "    response = graph.invoke({\"question\": user_question})\n",
        "    return response[\"answer\"].content\n",
        "\n",
        "# create a Gradio Interface\n",
        "chatbot_ui = gr.Interface(\n",
        "    fn=chatbot_response,\n",
        "    inputs=gr.Textbox(lines=2, placeholder=\"Ask a policy question...\"),\n",
        "    outputs=\"text\",\n",
        "    title=\"Policy Chatbot\",\n",
        "    description=\"Ask questions about internet commerce policies.\",\n",
        ")\n",
        "\n",
        "\n",
        "chatbot_ui.launch()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 648
        },
        "id": "2LpLqgdnRQqm",
        "outputId": "4d6c9a57-5175-4f9a-d0ab-8540b33e395c"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Running Gradio in a Colab notebook requires sharing enabled. Automatically setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n",
            "\n",
            "Colab notebook detected. To show errors in colab notebook, set debug=True in launch()\n",
            "* Running on public URL: https://3e81af64ac9330457c.gradio.live\n",
            "\n",
            "This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<div><iframe src=\"https://3e81af64ac9330457c.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": []
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip freeze > requirements.txt\n"
      ],
      "metadata": {
        "id": "dEFBD3C2CXH2"
      },
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "files.download(\"requirements.txt\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 17
        },
        "id": "oQMpffH4Cz5v",
        "outputId": "ea4150e6-fc63-4345-c314-fbfa1f6a3839"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "    async function download(id, filename, size) {\n",
              "      if (!google.colab.kernel.accessAllowed) {\n",
              "        return;\n",
              "      }\n",
              "      const div = document.createElement('div');\n",
              "      const label = document.createElement('label');\n",
              "      label.textContent = `Downloading \"${filename}\": `;\n",
              "      div.appendChild(label);\n",
              "      const progress = document.createElement('progress');\n",
              "      progress.max = size;\n",
              "      div.appendChild(progress);\n",
              "      document.body.appendChild(div);\n",
              "\n",
              "      const buffers = [];\n",
              "      let downloaded = 0;\n",
              "\n",
              "      const channel = await google.colab.kernel.comms.open(id);\n",
              "      // Send a message to notify the kernel that we're ready.\n",
              "      channel.send({})\n",
              "\n",
              "      for await (const message of channel.messages) {\n",
              "        // Send a message to notify the kernel that we're ready.\n",
              "        channel.send({})\n",
              "        if (message.buffers) {\n",
              "          for (const buffer of message.buffers) {\n",
              "            buffers.push(buffer);\n",
              "            downloaded += buffer.byteLength;\n",
              "            progress.value = downloaded;\n",
              "          }\n",
              "        }\n",
              "      }\n",
              "      const blob = new Blob(buffers, {type: 'application/binary'});\n",
              "      const a = document.createElement('a');\n",
              "      a.href = window.URL.createObjectURL(blob);\n",
              "      a.download = filename;\n",
              "      div.appendChild(a);\n",
              "      a.click();\n",
              "      div.remove();\n",
              "    }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "download(\"download_6bb1bd5d-9289-4c1c-a2b0-03fcab69a18f\", \"requirements.txt\", 13463)"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!echo \".env\" >> .gitignore\n",
        "!echo \"__pycache__/\" >> .gitignore\n",
        "!echo \"*.log\" >> .gitignore\n",
        "!echo \"*.zip\" >> .gitignore\n"
      ],
      "metadata": {
        "id": "SPS5qyYJamJ9"
      },
      "execution_count": 45,
      "outputs": []
    }
  ]
}