from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from pathlib import Path
from typing import List

class DocumentHandler:

    DATA_DIR = None

    def __init__(self, data_dir:str):
        self.DATA_DIR = Path(data_dir)

    def load_one_file(self, path: Path):
        suffix = path.suffix.lower()

        try:
            if suffix in {".txt"}:
                return TextLoader(str(path), autodetect_encoding=True).load()

            if suffix in {".md"}:
                try:
                    return UnstructuredMarkdownLoader(str(path), mode="elements").load()
                except Exception:
                    return TextLoader(str(path), autodetect_encoding=True).load()

            if suffix in {".docx"}:
                try:
                    return UnstructuredWordDocumentLoader(str(path), mode="elements").load()
                except Exception:
                    from langchain_community.document_loaders import Docx2txtLoader
                    return Docx2txtLoader(str(path)).load()

            if suffix in {".pdf"}:
                try:
                    return PyPDFLoader(str(path)).load()
                except Exception as e_pypdf:
                    try:
                        return UnstructuredPDFLoader(str(path), mode="elements").load()
                    except Exception as e_unstruct:
                        raise RuntimeError(
                            f"PDF failed with PyPDF and Unstructured: {e_pypdf} | {e_unstruct}"
                        )

            return TextLoader(str(path), autodetect_encoding=True).load()

        except Exception as e:
            raise RuntimeError(f"{path} → {type(e).__name__}: {e}") from e

    def load_documents(self) -> List:
        exts = (".txt", ".md", ".docx", ".pdf")
        files = [p for p in self.DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts]

        documents = []
        bad_files = []

        for fp in files:
            try:
                docs = self.load_one_file(fp)
                documents.extend(docs)
            except Exception as e:
                bad_files.append((fp, str(e)))
                print(f"[SKIP] {fp} — {e}")

        print(f"Loaded {len(documents)} docs. Skipped {len(bad_files)} bad files.")
        if bad_files:
            print("First few errors:")
            for fp, err in bad_files[:5]:
                print(f"  - {fp}: {err}")

        return documents

    def prepare_documents(self, documents):
        merged_docs = {}

        for doc in documents:
            if doc.metadata['source'].endswith('.docx') or doc.metadata['source'].endswith('.md'):
                if doc.metadata['source'] not in merged_docs:
                    merged_docs[doc.metadata['source']] = doc
                else:
                    merged_docs[doc.metadata['source']].page_content += "\n" + doc.page_content
            else:
                merged_docs[doc.metadata['source'] + f"_{doc.metadata['page'] if 'page' in doc.metadata else '0'}"] = doc
        return list(merged_docs.values())