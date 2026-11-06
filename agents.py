"""
Sistema Multi-Agente para Análise de Procedimentos Cirúrgicos com Ollama (ministral-3:3b).

Arquitetura da Equipa de Agentes:
1. AgentePesquisador   -> Triagem clínica, enquadramento e validação documental (RAG)
2. AgenteInstrumentacao -> Preparação pré-operatória, posicionamento e mesa de instrumentos
3. AgenteProcedimento   -> Sequência de etapas operatórias, técnica cirúrgica e segurança
4. AgenteRelator        -> Síntese executiva, auditoria clínica e consolidação do relatório
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import requests

from rag import BaseConhecimentoRAG, ResultadoPesquisaRAG


OLLAMA_URL_PADRAO = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODELO_PADRAO = os.environ.get("OLLAMA_MODEL", "ministral-3:3b")


class ErroConexaoOllama(Exception):
    """Exceção lançada quando o servidor Ollama ou o modelo não está acessível."""


class ClienteOllama:
    """
    Cliente HTTP com suporte a streaming para a API local do Ollama.
    O modo streaming evita timeouts em inferência local (CPU/GPU) porque recebe
    cada token à medida que é gerado em vez de bloquear até ao fim da resposta.
    """

    def __init__(
        self,
        modelo: str = MODELO_PADRAO,
        base_url: str = OLLAMA_URL_PADRAO,
        temperatura: float = 0.15,
        timeout_ligacao: int = 15,
        timeout_leitura_chunk: int = 300,
    ) -> None:
        self.modelo = modelo
        self.base_url = base_url.rstrip("/")
        self.temperatura = temperatura
        self.timeout_ligacao = timeout_ligacao
        self.timeout_leitura_chunk = timeout_leitura_chunk

    def verificar_estado(self) -> Tuple[bool, str, List[str]]:
        """
        Verifica se o servidor Ollama está ativo e se o modelo configurado existe.
        Retorna (sucesso, mensagem, lista_modelos_instalados).
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            dados = resp.json()
            modelos = [m.get("name", "") for m in dados.get("models", [])]

            encontrado = any(
                m == self.modelo or m.startswith(f"{self.modelo}:")
                for m in modelos
            )
            if not encontrado:
                return (
                    False,
                    f"O servidor Ollama está ativo, mas o modelo '{self.modelo}' não foi encontrado. "
                    f"Modelos disponíveis: {', '.join(modelos) if modelos else 'nenhum'}. "
                    f"Execute: ollama pull {self.modelo}",
                    modelos,
                )
            return True, f"Ollama ligado ({self.modelo})", modelos
        except requests.RequestException as exc:
            return (
                False,
                f"Não foi possível ligar ao Ollama em {self.base_url} ({exc}). "
                "Certifique-se de que o serviço está a correr ('ollama serve').",
                [],
            )

    def gerar_resposta(
        self,
        system_prompt: str,
        user_prompt: str,
        num_predict: int = 600,
        callback_token: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        """
        Envia uma mensagem para /api/chat do Ollama em modo STREAMING (`stream=True`).
        Isto evita ReadTimeout em computadores onde o carregamento do modelo ou a geração
        em CPU demora mais tempo, pois os pacotes chegam continuamente token a token.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.modelo,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "keep_alive": "15m",
            "options": {
                "temperature": self.temperatura,
                "num_predict": num_predict,
            },
        }

        partes_resposta: List[str] = []
        contador_chunks = 0

        try:
            with requests.post(
                url,
                json=payload,
                stream=True,
                timeout=(self.timeout_ligacao, self.timeout_leitura_chunk),
            ) as resp:
                resp.raise_for_status()
                for linha in resp.iter_lines(decode_unicode=True):
                    if not linha:
                        continue
                    dados = json.loads(linha)
                    if "error" in dados:
                        raise ErroConexaoOllama(f"Erro devolvido pelo Ollama: {dados['error']}")

                    pedaco = dados.get("message", {}).get("content", "")
                    if pedaco:
                        partes_resposta.append(pedaco)
                        contador_chunks += 1
                        if callback_token:
                            callback_token(pedaco, contador_chunks)

                    if dados.get("done", False):
                        break

            return "".join(partes_resposta).strip()
        except requests.RequestException as exc:
            raise ErroConexaoOllama(
                f"Erro ao comunicar com o Ollama ({self.modelo}) em {url}: {exc}"
            ) from exc


@dataclass
class RegistoEtapaAgente:
    """Guarda o resultado e métricas de execução de um agente individual."""
    nome_agente: str
    papel: str
    etapa_assegurada: str
    conteudo: str
    tempo_segundos: float
    tokens_gerados: int = 0


@dataclass
class RelatorioCirurgico:
    """Resultado completo produzido pela pipeline multi-agente."""
    pesquisa_utilizador: str
    data_geracao: str
    modelo_utilizado: str
    documentos_fonte: List[Dict[str, str]]
    etapas_agentes: List[RegistoEtapaAgente] = field(default_factory=list)
    markdown_final: str = ""
    caminho_ficheiro: Optional[str] = None


class AgenteBase:
    """Classe base para todos os agentes cirúrgicos especializados."""

    def __init__(
        self,
        nome: str,
        papel: str,
        etapa_assegurada: str,
        system_prompt: str,
        cliente: ClienteOllama,
        num_predict: int = 600,
    ) -> None:
        self.nome = nome
        self.papel = papel
        self.etapa_assegurada = etapa_assegurada
        self.system_prompt = system_prompt
        self.cliente = cliente
        self.num_predict = num_predict

    def executar(
        self,
        prompt_utilizador: str,
        callback_token: Optional[Callable[[str, int], None]] = None,
    ) -> RegistoEtapaAgente:
        """Executa o agente em streaming medindo o tempo e os tokens gerados."""
        inicio = time.perf_counter()
        total_tokens = 0

        def _on_token(pedaco: str, contagem: int) -> None:
            nonlocal total_tokens
            total_tokens = contagem
            if callback_token:
                callback_token(pedaco, contagem)

        resposta = self.cliente.gerar_resposta(
            system_prompt=self.system_prompt,
            user_prompt=prompt_utilizador,
            num_predict=self.num_predict,
            callback_token=_on_token,
        )
        duracao = time.perf_counter() - inicio
        return RegistoEtapaAgente(
            nome_agente=self.nome,
            papel=self.papel,
            etapa_assegurada=self.etapa_assegurada,
            conteudo=resposta,
            tempo_segundos=duracao,
            tokens_gerados=total_tokens,
        )


class EquipaAgentesCirurgicos:
    """
    Orquestra os 4 agentes especializados sobre a base de conhecimento RAG em `docs/`.
    Cada agente assegura uma etapa distinta do procedimento cirúrgico.
    """

    def __init__(
        self,
        rag: BaseConhecimentoRAG,
        cliente: ClienteOllama,
    ) -> None:
        self.rag = rag
        self.cliente = cliente

        # 1. Agente Pesquisador e Triagem Pré-Operatória
        self.agente_pesquisador = AgenteBase(
            nome="Agente 1 — Triagem e Enquadramento Clínico (RAG)",
            papel="Especialista em Documentação Cirúrgica e Triagem Pré-Operatória",
            etapa_assegurada="Etapa 1: Identificação do Procedimento, Indicações e Anestesia",
            system_prompt=(
                "És o Agente de Triagem Cirúrgica. Responde em Português de Portugal (PT-PT) de forma direta e concisa.\n"
                "Usa APENAS os dados do documento fornecido:\n"
                "- Procedimento, código, especialidade, duração e anestesia;\n"
                "- Indicações cirúrgicas principais."
            ),
            cliente=self.cliente,
            num_predict=350,
        )

        # 2. Agente de Preparação e Instrumentação Cirúrgica
        self.agente_instrumentacao = AgenteBase(
            nome="Agente 2 — Preparação e Instrumentação Cirúrgica",
            papel="Enfermeiro Instrumentista Chefe / Especialista em Mesa Operatória",
            etapa_assegurada="Etapa 2: Preparação Pré-Operatória, Posicionamento e Instrumentos",
            system_prompt=(
                "És o Agente de Instrumentação Cirúrgica. Responde em Português de Portugal (PT-PT) de forma estruturada.\n"
                "Com base exclusiva no documento fornecido, apresenta:\n"
                "1. Preparação pré-operatória e posicionamento do doente;\n"
                "2. Lista organizada por categorias de todos os instrumentos cirúrgicos, equipamentos e fios de sutura necessários."
            ),
            cliente=self.cliente,
            num_predict=650,
        )

        # 3. Agente de Procedimento e Etapas Operatórias
        self.agente_procedimento = AgenteBase(
            nome="Agente 3 — Condução Técnica e Etapas Operatórias",
            papel="Cirurgião Principal / Coordenador dos Tempos Operatórios",
            etapa_assegurada="Etapa 3: Execução Passo a Passo das Etapas Cirúrgicas e Segurança",
            system_prompt=(
                "És o Agente Cirurgião responsável pelas etapas operatórias. Responde em Português de Portugal (PT-PT).\n"
                "Com base exclusiva no documento fornecido, descreve de forma clara:\n"
                "1. As etapas cronológicas da cirurgia passo a passo (indicando os instrumentos usados);\n"
                "2. Os cuidados críticos de segurança intraoperatória."
            ),
            cliente=self.cliente,
            num_predict=700,
        )

        # 4. Agente Relator e Auditor Clínico
        self.agente_relator = AgenteBase(
            nome="Agente 4 — Auditoria e Síntese Executiva",
            papel="Auditor Clínico e Redator de Relatórios Cirúrgicos",
            etapa_assegurada="Etapa 4: Síntese Executiva e Checklist de Verificação Operatória",
            system_prompt=(
                "És o Agente Auditor Clínico. Responde em Português de Portugal (PT-PT) de forma concisa.\n"
                "Redige:\n"
                "1. Uma síntese executiva curta (1 parágrafo) que responda à pesquisa do utilizador;\n"
                "2. Uma Checklist de Verificação Operatória com 5 a 7 pontos essenciais (pré, intra e pós-operatório)."
            ),
            cliente=self.cliente,
            num_predict=400,
        )

    def processar_pesquisa(
        self,
        pesquisa_utilizador: str,
        callback_progresso: Optional[Callable[[int, int, str, str], None]] = None,
        callback_token_agente: Optional[Callable[[int, int, str, str, int], None]] = None,
        callback_conclusao_agente: Optional[Callable[[RegistoEtapaAgente], None]] = None,
    ) -> RelatorioCirurgico:
        """
        Executa a pesquisa RAG e a sequência coordenada dos 4 agentes cirúrgicos.
        """
        resultado_rag: ResultadoPesquisaRAG = self.rag.pesquisar(pesquisa_utilizador)

        docs_info = [
            {
                "id": doc.id,
                "ficheiro": f"docs/{doc.ficheiro}",
                "titulo": doc.titulo,
                "especialidade": doc.especialidade,
                "codigo": doc.codigo,
                "score": f"{score:.2f}",
            }
            for doc, score in resultado_rag.documentos_relevantes
        ]

        etapas_executadas: List[RegistoEtapaAgente] = []

        # ---------------------------------------------------------------------
        # AGENTE 1: Triagem e Enquadramento Clínico
        # ---------------------------------------------------------------------
        if callback_progresso:
            callback_progresso(
                1, 4, self.agente_pesquisador.nome, self.agente_pesquisador.etapa_assegurada
            )

        prompt_ag1 = (
            f"PESQUISA: \"{pesquisa_utilizador}\"\n\n"
            f"DOCUMENTO FONTE:\n{resultado_rag.contexto_para_pesquisador()}\n\n"
            "Apresenta o enquadramento clínico (código, especialidade, duração, anestesia) e as indicações cirúrgicas."
        )
        registo_ag1 = self.agente_pesquisador.executar(
            prompt_ag1,
            callback_token=(
                (lambda tok, n: callback_token_agente(1, 4, self.agente_pesquisador.nome, tok, n))
                if callback_token_agente
                else None
            ),
        )
        etapas_executadas.append(registo_ag1)
        if callback_conclusao_agente:
            callback_conclusao_agente(registo_ag1)

        # ---------------------------------------------------------------------
        # AGENTE 2: Preparação Pré-Operatória e Instrumentação
        # ---------------------------------------------------------------------
        if callback_progresso:
            callback_progresso(
                2, 4, self.agente_instrumentacao.nome, self.agente_instrumentacao.etapa_assegurada
            )

        prompt_ag2 = (
            f"PESQUISA: \"{pesquisa_utilizador}\"\n\n"
            f"DOCUMENTO FONTE (PREPARAÇÃO E INSTRUMENTOS):\n"
            f"{resultado_rag.contexto_para_instrumentacao()}\n\n"
            "Lista a preparação pré-operatória, posicionamento e todos os instrumentos cirúrgicos e suturas por categoria."
        )
        registo_ag2 = self.agente_instrumentacao.executar(
            prompt_ag2,
            callback_token=(
                (lambda tok, n: callback_token_agente(2, 4, self.agente_instrumentacao.nome, tok, n))
                if callback_token_agente
                else None
            ),
        )
        etapas_executadas.append(registo_ag2)
        if callback_conclusao_agente:
            callback_conclusao_agente(registo_ag2)

        # ---------------------------------------------------------------------
        # AGENTE 3: Etapas do Procedimento Cirúrgico e Segurança
        # ---------------------------------------------------------------------
        if callback_progresso:
            callback_progresso(
                3, 4, self.agente_procedimento.nome, self.agente_procedimento.etapa_assegurada
            )

        prompt_ag3 = (
            f"PESQUISA: \"{pesquisa_utilizador}\"\n\n"
            f"DOCUMENTO FONTE (ETAPAS E SEGURANÇA):\n"
            f"{resultado_rag.contexto_para_procedimento()}\n\n"
            "Descreve passo a passo todas as etapas cirúrgicas e os cuidados críticos de segurança."
        )
        registo_ag3 = self.agente_procedimento.executar(
            prompt_ag3,
            callback_token=(
                (lambda tok, n: callback_token_agente(3, 4, self.agente_procedimento.nome, tok, n))
                if callback_token_agente
                else None
            ),
        )
        etapas_executadas.append(registo_ag3)
        if callback_conclusao_agente:
            callback_conclusao_agente(registo_ag3)

        # ---------------------------------------------------------------------
        # AGENTE 4: Síntese Executiva e Auditoria Final
        # ---------------------------------------------------------------------
        if callback_progresso:
            callback_progresso(
                4, 4, self.agente_relator.nome, self.agente_relator.etapa_assegurada
            )

        resumo_fontes = "\n".join(
            f"- {d.titulo} ({d.especialidade}, {d.duracao}, {d.anestesia})"
            for d, _ in resultado_rag.documentos_relevantes
        )
        prompt_ag4 = (
            f"PESQUISA DO UTILIZADOR: \"{pesquisa_utilizador}\"\n"
            f"PROCEDIMENTO(S): {resumo_fontes}\n\n"
            f"RESUMO TRIAGEM:\n{registo_ag1.conteudo[:800]}\n\n"
            f"SEGURANÇA E PASSOS CHAVE:\n{registo_ag3.conteudo[-800:]}\n\n"
            "Elabora uma Síntese Executiva concisa e uma Checklist de Verificação Operatória (5 a 7 itens)."
        )
        registo_ag4 = self.agente_relator.executar(
            prompt_ag4,
            callback_token=(
                (lambda tok, n: callback_token_agente(4, 4, self.agente_relator.nome, tok, n))
                if callback_token_agente
                else None
            ),
        )
        etapas_executadas.append(registo_ag4)
        if callback_conclusao_agente:
            callback_conclusao_agente(registo_ag4)

        data_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        relatorio = RelatorioCirurgico(
            pesquisa_utilizador=pesquisa_utilizador,
            data_geracao=data_str,
            modelo_utilizado=self.cliente.modelo,
            documentos_fonte=docs_info,
            etapas_agentes=etapas_executadas,
        )
        relatorio.markdown_final = self._compilar_markdown_relatorio(relatorio)
        return relatorio

    def _compilar_markdown_relatorio(self, relatorio: RelatorioCirurgico) -> str:
        """Compila o documento Markdown final integrando o trabalho dos 4 agentes."""
        tempo_total = sum(e.tempo_segundos for e in relatorio.etapas_agentes)
        fontes_md = "\n".join(
            f"- **{d['titulo']}** (`{d['ficheiro']}`) — Especialidade: *{d['especialidade']}* | "
            f"Código: `{d['codigo']}` | Relevância RAG: `{d['score']}`"
            for d in relatorio.documentos_fonte
        )

        ag1, ag2, ag3, ag4 = relatorio.etapas_agentes

        return (
            f"# Relatório Técnico de Procedimento Cirúrgico\n\n"
            f"- **Pesquisa do Utilizador:** {relatorio.pesquisa_utilizador}\n"
            f"- **Data de Geração:** {relatorio.data_geracao}\n"
            f"- **Modelo LLM (Ollama):** `{relatorio.modelo_utilizado}`\n"
            f"- **Tempo Total de Processamento:** {tempo_total:.1f}s\n\n"
            f"## Fontes Documentais Consultadas (`docs/`)\n"
            f"{fontes_md}\n\n"
            f"---\n\n"
            f"## 1. Síntese Executiva e Checklist de Auditoria\n"
            f"> *Assegurado por: **{ag4.nome_agente}** ({ag4.tempo_segundos:.1f}s)*\n\n"
            f"{ag4.conteudo}\n\n"
            f"---\n\n"
            f"## 2. Triagem Clínica e Indicações Operatórias\n"
            f"> *Assegurado por: **{ag1.nome_agente}** ({ag1.tempo_segundos:.1f}s)*\n\n"
            f"{ag1.conteudo}\n\n"
            f"---\n\n"
            f"## 3. Preparação Pré-Operatória, Posicionamento e Instrumentação\n"
            f"> *Assegurado por: **{ag2.nome_agente}** ({ag2.tempo_segundos:.1f}s)*\n\n"
            f"{ag2.conteudo}\n\n"
            f"---\n\n"
            f"## 4. Etapas do Procedimento Cirúrgico e Segurança Intraoperatória\n"
            f"> *Assegurado por: **{ag3.nome_agente}** ({ag3.tempo_segundos:.1f}s)*\n\n"
            f"{ag3.conteudo}\n"
        )
