"""
Motor de RAG (Retrieval-Augmented Generation) Local para Documentos Cirúrgicos.

Carrega os documentos da pasta `docs/`, divide-os por secções operatórias
(Indicações, Preparação/Posicionamento, Instrumentos, Etapas Cirúrgicas e Segurança)
e disponibiliza pesquisa híbrida (BM25 + normalização linguística PT + expansão de
sinónimos clínicos) para fornecer contexto preciso a cada agente especializado.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Stopwords comuns em português para filtragem na indexação
STOPWORDS_PT = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as",
    "ate", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles",
    "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre",
    "era", "eram", "essa", "essas", "esse", "esses", "esta", "estas", "este",
    "estes", "eu", "foi", "for", "foram", "ha", "isso", "isto", "ja", "la",
    "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha",
    "minhas", "muito", "na", "nas", "nao", "nem", "no", "nos", "nossa",
    "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelas", "pelo", "pelos", "por", "qual", "quais", "quando",
    "que", "quem", "sao", "se", "sem", "seu", "seus", "so", "sua", "suas",
    "tambem", "te", "tem", "tendo", "tenho", "ter", "teu", "teus", "ti",
    "tinha", "tinham", "todo", "todos", "tu", "tua", "tuas", "um", "uma",
    "umas", "uns", "vai", "ir", "ser", "estar", "sobre", "quaisquer",
}

# Mapa de sinónimos clínicos e termos coloquiais para enriquecer a pesquisa
SINONIMOS_CLINICOS: Dict[str, List[str]] = {
    "apendice": ["apendicectomia", "apendicite", "apendicular", "ileocecal", "fossa", "iliaca"],
    "apendicectomia": ["apendice", "apendicite", "apendicular", "endoloop", "trocarte"],
    "apendicite": ["apendicectomia", "apendice", "apendicular", "endoloop", "trocarte"],
    "vesicula": ["colecistectomia", "biliar", "cistico", "calot", "strasberg", "litiase", "calculos"],
    "colecistectomia": ["vesicula", "biliar", "cistico", "calot", "strasberg", "litiase"],
    "biliar": ["colecistectomia", "vesicula", "cistico", "calot", "strasberg", "colangiografia"],
    "colecistite": ["colecistectomia", "vesicula", "biliar", "calot", "strasberg"],
    "anca": ["artroplastia", "coxartrose", "femoral", "acetabular", "femur", "protese", "ortopedia"],
    "artroplastia": ["anca", "coxartrose", "femoral", "acetabular", "femur", "protese"],
    "femur": ["artroplastia", "anca", "femoral", "acetabular", "trocanter", "osteotomia"],
    "coxartrose": ["artroplastia", "anca", "acetabular", "femoral", "charnley", "hohmann"],
    "hernia": ["hernioplastia", "inguinal", "lichtenstein", "polipropileno", "rede", "espermático"],
    "hernioplastia": ["hernia", "inguinal", "lichtenstein", "polipropileno", "rede", "poupart"],
    "virilha": ["hernioplastia", "inguinal", "lichtenstein", "hernia", "poupart"],
    "inguinal": ["hernioplastia", "lichtenstein", "hernia", "poupart", "cremaster"],
    "cerebro": ["craniotomia", "neurocirurgia", "subdural", "hematoma", "dura", "trepano", "raney"],
    "craniotomia": ["cranio", "subdural", "hematoma", "neurocirurgia", "dura", "trepano", "raney"],
    "cranio": ["craniotomia", "subdural", "hematoma", "trepano", "craniotomo", "mayfield"],
    "hematoma": ["craniotomia", "subdural", "drenagem", "neurocirurgia", "dura", "raney"],
    "subdural": ["craniotomia", "hematoma", "neurocirurgia", "trepano", "fukushima"],
    "coracao": ["revascularizacao", "miocardio", "cabg", "bypass", "coronario", "esternotomia", "cec"],
    "revascularizacao": ["miocardio", "cabg", "bypass", "coronario", "esternotomia", "cec"],
    "coronaria": ["revascularizacao", "miocardio", "cabg", "bypass", "mamaria", "safena", "esternotomia"],
    "bypass": ["revascularizacao", "miocardio", "cabg", "coronaria", "esternotomia", "cardioplegia"],
    "ponte": ["revascularizacao", "miocardio", "cabg", "bypass", "coronaria", "safena"],
    "parto": ["cesariana", "obstetricia", "pfannenstiel", "kerr", "histerotomia", "fetal", "utero"],
    "gravida": ["cesariana", "obstetricia", "pfannenstiel", "kerr", "histerotomia", "fetal"],
    "bebe": ["cesariana", "fetal", "umbilical", "pfannenstiel", "kerr", "obstetricia"],
    "utero": ["cesariana", "histerotomia", "histerorrafia", "pfannenstiel", "kerr", "ocitocina"],
    "cesariana": ["pfannenstiel", "kerr", "histerotomia", "histerorrafia", "doyen", "obstetricia"],
}


def normalizar_texto(texto: str) -> str:
    """Remove acentos, converte para minúsculas e limpa pontuação."""
    sem_acentos = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )
    return sem_acentos


def tokenizar(texto: str, remover_stopwords: bool = True) -> List[str]:
    """Extrai tokens alfanuméricos normalizados de um texto."""
    norm = normalizar_texto(texto)
    tokens = re.findall(r"[a-z0-9]{2,}", norm)
    if remover_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_PT]
    return tokens


def radical_simples(token: str) -> str:
    """Reduz sufixos comuns em português para melhorar correspondência lexical."""
    if len(token) <= 4:
        return token
    for sufixo in ("ectomias", "ectomia", "plastias", "plastia", "tomias", "tomia",
                   "scopica", "scopico", "amento", "imentos", "imento", "acoes",
                   "acao", "icos", "icas", "ico", "ica", "ais", "oes", "es", "os", "as", "s"):
        if token.endswith(sufixo) and len(token) - len(sufixo) >= 4:
            return token[: -len(sufixo)]
    return token


@dataclass
class SeccaoDocumento:
    """Representa uma secção temática dentro de um documento cirúrgico."""
    doc_id: str
    doc_titulo: str
    ficheiro: str
    titulo_seccao: str
    categoria: str  # 'indicacoes', 'preparacao', 'instrumentos', 'etapas', 'seguranca', 'geral'
    conteudo: str
    tokens: List[str] = field(default_factory=list)


@dataclass
class DocumentoCirurgico:
    """Representa um documento completo de procedimento cirúrgico na pasta docs/."""
    id: str
    ficheiro: str
    caminho: Path
    titulo: str
    especialidade: str
    codigo: str
    duracao: str
    anestesia: str
    conteudo_completo: str
    seccoes: List[SeccaoDocumento] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)

    def obter_seccoes_por_categoria(self, categorias: List[str]) -> str:
        """Devolve o texto concatenado das secções pertencentes às categorias pedidas."""
        blocos = [
            f"## {sec.titulo_seccao}\n{sec.conteudo}"
            for sec in self.seccoes
            if sec.categoria in categorias
        ]
        if blocos:
            return "\n\n".join(blocos)
        return self.conteudo_completo


@dataclass
class ResultadoPesquisaRAG:
    """Resultado estruturado de uma consulta à base de conhecimento RAG."""
    query_original: str
    termos_expandidos: List[str]
    documentos_relevantes: List[Tuple[DocumentoCirurgico, float]]
    seccoes_relevantes: List[Tuple[SeccaoDocumento, float]]

    def contexto_para_pesquisador(self) -> str:
        """Gera um resumo documental global para o Agente Pesquisador."""
        if not self.documentos_relevantes:
            return "Nenhum documento relevante encontrado na pasta docs/."

        partes = []
        for doc, score in self.documentos_relevantes:
            seccoes_nomes = ", ".join(s.titulo_seccao for s in doc.seccoes)
            indicacoes = doc.obter_seccoes_por_categoria(["indicacoes", "preparacao"])
            partes.append(
                f"=== DOCUMENTO: {doc.titulo} (Ficheiro: {doc.ficheiro} | Relevância: {score:.2f}) ===\n"
                f"Especialidade: {doc.especialidade} | Código: {doc.codigo}\n"
                f"Duração Estimada: {doc.duracao} | Anestesia: {doc.anestesia}\n"
                f"Secções Disponíveis: {seccoes_nomes}\n\n"
                f"{indicacoes}"
            )
        return "\n\n" + ("\n" + "=" * 70 + "\n\n").join(partes)

    def contexto_para_instrumentacao(self) -> str:
        """Fornece as secções de preparação, posicionamento e instrumentos para o Agente de Instrumentação."""
        if not self.documentos_relevantes:
            return "Nenhum documento relevante encontrado."

        partes = []
        for doc, _ in self.documentos_relevantes:
            conteudo_inst = doc.obter_seccoes_por_categoria(["preparacao", "instrumentos"])
            partes.append(
                f"=== PROCEDIMENTO: {doc.titulo} (Fonte: docs/{doc.ficheiro}) ===\n"
                f"Especialidade: {doc.especialidade} | Anestesia: {doc.anestesia}\n\n"
                f"{conteudo_inst}"
            )
        return "\n\n".join(partes)

    def contexto_para_procedimento(self) -> str:
        """Fornece as secções de indicações, etapas cirúrgicas e segurança para o Agente de Procedimento."""
        if not self.documentos_relevantes:
            return "Nenhum documento relevante encontrado."

        partes = []
        for doc, _ in self.documentos_relevantes:
            conteudo_etapas = doc.obter_seccoes_por_categoria(["indicacoes", "etapas", "seguranca"])
            partes.append(
                f"=== PROCEDIMENTO: {doc.titulo} (Fonte: docs/{doc.ficheiro}) ===\n"
                f"Especialidade: {doc.especialidade} | Duração: {doc.duracao}\n\n"
                f"{conteudo_etapas}"
            )
        return "\n\n".join(partes)


class BaseConhecimentoRAG:
    """Gere o carregamento, indexação e pesquisa sobre os documentos cirúrgicos em docs/."""

    def __init__(self, pasta_docs: Path | str = "docs") -> None:
        self.pasta_docs = Path(pasta_docs)
        self.documentos: List[DocumentoCirurgico] = []
        self.seccoes: List[SeccaoDocumento] = []
        self.carregar_documentos()

    def carregar_documentos(self) -> int:
        """Carrega e processa todos os ficheiros .md e .txt da pasta docs/."""
        self.documentos.clear()
        self.seccoes.clear()

        if not self.pasta_docs.exists():
            self.pasta_docs.mkdir(parents=True, exist_ok=True)
            return 0

        ficheiros = sorted(
            list(self.pasta_docs.glob("*.md")) + list(self.pasta_docs.glob("*.txt"))
        )

        for idx, caminho in enumerate(ficheiros, start=1):
            texto = caminho.read_text(encoding="utf-8").strip()
            if not texto:
                continue
            doc = self._parse_documento(idx, caminho, texto)
            self.documentos.append(doc)
            self.seccoes.extend(doc.seccoes)

        return len(self.documentos)

    def _classificar_categoria_seccao(self, titulo_seccao: str) -> str:
        """Classifica o tipo de secção com base no seu título."""
        norm = normalizar_texto(titulo_seccao)
        if any(p in norm for p in ("instrument", "equipament", "material", "mesa", "protese")):
            return "instrumentos"
        if any(p in norm for p in ("etapa", "passo", "procedimento", "tecnica", "tempo", "operat")):
            if "pre-operat" in norm or "preparac" in norm or "posicionamento" in norm:
                return "preparacao"
            return "etapas"
        if any(p in norm for p in ("preparac", "posicionamento", "pre-operat", "anestesi")):
            return "preparacao"
        if any(p in norm for p in ("indicac", "diagnost", "objetivo")):
            return "indicacoes"
        if any(p in norm for p in ("cuidado", "seguranca", "critico", "complicac", "checklist")):
            return "seguranca"
        return "geral"

    def _parse_documento(self, idx: int, caminho: Path, texto: str) -> DocumentoCirurgico:
        """Extrai cabeçalho, metadados e secções estruturadas de um documento."""
        linhas = texto.splitlines()
        titulo = caminho.stem.replace("_", " ").title()

        for linha in linhas:
            if linha.strip().startswith("# "):
                titulo = linha.strip()[2:].strip()
                break

        def extrair_meta(padrao: str, valor_omissao: str = "Não especificado") -> str:
            match = re.search(padrao, texto, flags=re.IGNORECASE)
            return match.group(1).strip() if match else valor_omissao

        especialidade = extrair_meta(r"\*\*Especialidade:\*\*\s*(.+)")
        codigo = extrair_meta(r"\*\*C[óo]digo do Procedimento:\*\*\s*(.+)", f"DOC-{idx:03d}")
        duracao = extrair_meta(r"\*\*Dura[çc][ãa]o Estimada:\*\*\s*(.+)")
        anestesia = extrair_meta(r"\*\*Tipo de Anestesia:\*\*\s*(.+)")

        doc_id = f"DOC-{idx:02d}"

        # Dividir por secções de nível 2 (## )
        partes_h2 = re.split(r"(?m)^##\s+", texto)
        seccoes: List[SeccaoDocumento] = []

        for parte in partes_h2[1:]:
            linhas_sec = parte.strip().splitlines()
            if not linhas_sec:
                continue
            titulo_sec = linhas_sec[0].strip()
            conteudo_sec = "\n".join(linhas_sec[1:]).strip()
            categoria = self._classificar_categoria_seccao(titulo_sec)
            tokens_sec = tokenizar(f"{titulo} {especialidade} {titulo_sec} {conteudo_sec}")
            seccoes.append(
                SeccaoDocumento(
                    doc_id=doc_id,
                    doc_titulo=titulo,
                    ficheiro=caminho.name,
                    titulo_seccao=titulo_sec,
                    categoria=categoria,
                    conteudo=conteudo_sec,
                    tokens=tokens_sec,
                )
            )

        # Caso o documento não tenha cabeçalhos ##, criar uma secção única geral
        if not seccoes:
            seccoes.append(
                SeccaoDocumento(
                    doc_id=doc_id,
                    doc_titulo=titulo,
                    ficheiro=caminho.name,
                    titulo_seccao="Procedimento Completo",
                    categoria="geral",
                    conteudo=texto,
                    tokens=tokenizar(texto),
                )
            )

        tokens_doc = tokenizar(texto)
        return DocumentoCirurgico(
            id=doc_id,
            ficheiro=caminho.name,
            caminho=caminho,
            titulo=titulo,
            especialidade=especialidade,
            codigo=codigo,
            duracao=duracao,
            anestesia=anestesia,
            conteudo_completo=texto,
            seccoes=seccoes,
            tokens=tokens_doc,
        )

    def expandir_query(self, query: str) -> List[str]:
        """Expande os termos da pesquisa com radicais e sinónimos clínicos."""
        tokens_base = tokenizar(query)
        expandidos = list(tokens_base)

        for t in tokens_base:
            rad = radical_simples(t)
            for chave, lista_sin in SINONIMOS_CLINICOS.items():
                if t == chave or rad == radical_simples(chave) or chave in t:
                    for s in lista_sin:
                        s_norm = normalizar_texto(s)
                        if s_norm not in expandidos:
                            expandidos.append(s_norm)
        return expandidos

    def _calcular_bm25(
        self, query_tokens: List[str], lista_tokens_docs: List[List[str]], k1: float = 1.5, b: float = 0.75
    ) -> List[float]:
        """Calcula pontuações BM25 combinadas com correspondência de radicais."""
        n_docs = len(lista_tokens_docs)
        if n_docs == 0:
            return []

        avgdl = sum(len(d) for d in lista_tokens_docs) / max(n_docs, 1)
        radicais_docs = [[radical_simples(t) for t in doc_t] for doc_t in lista_tokens_docs]
        scores = [0.0] * n_docs

        for q in query_tokens:
            q_rad = radical_simples(q)
            # Número de documentos que contêm o termo ou o seu radical
            df = sum(
                1 for i in range(n_docs)
                if (q in lista_tokens_docs[i] or q_rad in radicais_docs[i])
            )
            if df == 0:
                continue

            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

            for i in range(n_docs):
                tf_exato = lista_tokens_docs[i].count(q)
                tf_radical = radicais_docs[i].count(q_rad)
                tf = tf_exato + 0.6 * max(0, tf_radical - tf_exato)
                if tf > 0:
                    dl = len(lista_tokens_docs[i])
                    numerador = tf * (k1 + 1.0)
                    denominador = tf + k1 * (1.0 - b + b * (dl / max(avgdl, 1.0)))
                    scores[i] += idf * (numerador / denominador)

        return scores

    def pesquisar(self, query: str, top_k_docs: int = 2, top_k_seccoes: int = 6) -> ResultadoPesquisaRAG:
        """
        Realiza pesquisa híbrida sobre os documentos e secções cirúrgicas.
        Se o utilizador pedir uma comparação ou visão geral de várias cirurgias,
        adapta o número de documentos retornados.
        """
        if not self.documentos:
            return ResultadoPesquisaRAG(query, [], [], [])

        tokens_originais = tokenizar(query)
        termos_expandidos = self.expandir_query(query)

        # Pontuação dos documentos completos
        lista_tokens_docs = [d.tokens for d in self.documentos]
        scores_bm25_docs = self._calcular_bm25(termos_expandidos, lista_tokens_docs)

        docs_pontuados: List[Tuple[DocumentoCirurgico, float]] = []
        for doc, score in zip(self.documentos, scores_bm25_docs):
            titulo_norm = normalizar_texto(f"{doc.titulo} {doc.especialidade} {doc.codigo}")
            bonus_titulo = 0.0
            for t in tokens_originais:
                if t in titulo_norm or radical_simples(t) in titulo_norm:
                    bonus_titulo += 4.5
            for t in termos_expandidos:
                if t in titulo_norm:
                    bonus_titulo += 1.5

            score_final = score + bonus_titulo
            if score_final > 0:
                docs_pontuados.append((doc, score_final))

        docs_pontuados.sort(key=lambda x: x[1], reverse=True)

        # Se nenhum documento teve match direto (ex: pergunta muito genérica "que cirurgias existem?"),
        # devolve os documentos principais para que o agente possa listar/responder
        if not docs_pontuados:
            docs_pontuados = [(d, 1.0) for d in self.documentos[: max(top_k_docs, 3)]]
        else:
            # Se o 1.º documento tem um score muito superior ao 2.º (pesquisa focada numa única cirurgia),
            # focar apenas nesse documento principal para não poluir o contexto de um modelo 3B!
            if len(docs_pontuados) > 1 and docs_pontuados[0][1] >= docs_pontuados[1][1] * 1.65:
                docs_pontuados = [docs_pontuados[0]]
            else:
                docs_pontuados = docs_pontuados[:top_k_docs]

        # Pontuação das secções dentro dos documentos selecionados
        ids_selecionados = {d.id for d, _ in docs_pontuados}
        seccoes_candidatas = [s for s in self.seccoes if s.doc_id in ids_selecionados]
        scores_sec = self._calcular_bm25(
            termos_expandidos, [s.tokens for s in seccoes_candidatas]
        )

        seccoes_pontuadas = sorted(
            zip(seccoes_candidatas, scores_sec),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k_seccoes]

        return ResultadoPesquisaRAG(
            query_original=query,
            termos_expandidos=termos_expandidos,
            documentos_relevantes=docs_pontuados,
            seccoes_relevantes=seccoes_pontuadas,
        )
