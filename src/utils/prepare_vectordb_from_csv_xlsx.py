import os
import pandas as pd
from utils.load_config import LoadConfig
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings


class PrepareVectorDBFromTabularData:

    def __init__(self, file_directory: str) -> None:
        
        self.APPCFG = LoadConfig()
        self.file_directory = file_directory

    def run_pipeline(self):
        
        self.df, self.file_name = self._load_dataframe(self.file_directory)
        self.docs, self.metadatas, self.ids = self._prepare_data_for_injection(self.df, self.file_name)
        embeddings = self.APPCFG.embedding_model.embed_documents(self.docs)
        self.embeddings = [np.array(e, dtype=np.float32) for e in embeddings]
        self._inject_data_into_chromadb()
        self._validate_db()

    def _inject_data_into_chromadb(self):
       
        collection = self.APPCFG.chroma_client.create_collection(name=self.APPCFG.collection_name)
        collection.add(
            documents=self.docs,
            metadatas=self.metadatas,
            embeddings=self.embeddings,
            ids=self.ids
        )
        print("==============================")
        print("Data is stored in ChromaDB.")

    def _load_dataframe(self, file_directory: str):
        
        file_name_ext = os.path.basename(file_directory)
        file_name, file_extension = os.path.splitext(file_name_ext)
        if file_extension == ".csv":
            df = pd.read_csv(file_directory)
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_directory)
        else:
            raise ValueError("The selected file type is not supported")
        return df, file_name

    def _prepare_data_for_injection(self, df: pd.DataFrame, file_name: str):
        
        docs, metadatas, ids = [], [], []
        for idx, row in df.iterrows():
            row_str = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            docs.append(row_str)
            metadatas.append({"source": file_name})
            ids.append(f"id{idx}")
        return docs, metadatas, ids

        # embeddings = self.APPCFG.embedding_model.embed_documents(docs)

        # return docs, metadatas, ids, embeddings

    def _validate_db(self):
        vectordb = self.APPCFG.chroma_client.get_collection(
            name=self.APPCFG.collection_name
        )
        print("==============================")
        print("Number of vectors in vectordb:", vectordb.count())
        print("==============================")
