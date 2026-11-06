#!/usr/bin/env python3
"""
Aplicação de Terminal — Sistema Multi-Agente de Relatórios Cirúrgicos (RAG + Ollama).

Utiliza o modelo local `ministral-3:3b` via Ollama e documentos na pasta `docs/`
para gerar relatórios estruturados sobre procedimentos cirúrgicos, assegurando
cada etapa operatória através de agentes especializados.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.text import Text

from agents import (
    MODELO_PADRAO,
    OLLAMA_URL_PADRAO,
    ClienteOllama,
    EquipaAgentesCirurgicos,
    ErroConexaoOllama,
    RegistoEtapaAgente,
    RelatorioCirurgico,
)
from rag import BaseConhecimentoRAG, normalizar_texto


console = Console()


def criar_slug(texto: str, max_len: int = 45) -> str:
    """Converte uma pesquisa num nome de ficheiro seguro."""
    norm = normalizar_texto(texto)
    slug = re.sub(r"[^a-z0-9]+", "_", norm).strip("_")
    return slug[:max_len].rstrip("_") or "relatorio_cirurgico"


def guardar_relatorio(relatorio: RelatorioCirurgico, pasta_relatorios: Path) -> Path:
    """Guarda o relatório final em formato Markdown na pasta `relatorios/`."""
    pasta_relatorios.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = criar_slug(relatorio.pesquisa_utilizador)
    caminho = pasta_relatorios / f"relatorio_{timestamp}_{slug}.md"
    caminho.write_text(relatorio.markdown_final, encoding="utf-8")
    relatorio.caminho_ficheiro = str(caminho)
    return caminho


def mostrar_cabecalho(
    rag: BaseConhecimentoRAG,
    cliente: ClienteOllama,
    estado_ok: bool,
    msg_estado: str,
) -> None:
    """Apresenta o painel principal da aplicação no terminal."""
    cor_estado = "bold green" if estado_ok else "bold yellow"
    icone_estado = "ONLINE" if estado_ok else "AVISO"

    texto = Text()
    texto.append("Equipa Multi-Agente de Apoio a Procedimentos Cirúrgicos (RAG)\n", style="bold cyan")
    texto.append("Modelo LLM: ", style="dim")
    texto.append(f"{cliente.modelo} ", style="bold white")
    texto.append(f"({cliente.base_url})\n", style="dim")
    texto.append("Estado Ollama: ", style="dim")
    texto.append(f"[{icone_estado}] {msg_estado}\n", style=cor_estado)
    texto.append("Base Documental (`docs/`): ", style="dim")
    texto.append(
        f"{len(rag.documentos)} procedimentos cirúrgicos indexados ({len(rag.seccoes)} secções)\n\n",
        style="bold green",
    )
    texto.append("Pipeline de Agentes Especializados:\n", style="bold magenta")
    texto.append("  1. Agente Pesquisador (RAG)    -> Triagem clínica, indicações e enquadramento\n")
    texto.append("  2. Agente de Instrumentação    -> Preparação, posicionamento e mesa cirúrgica\n")
    texto.append("  3. Agente de Procedimento      -> Etapas operatórias passo a passo e segurança\n")
    texto.append("  4. Agente Relator & Auditor    -> Síntese executiva, checklist e relatório final")

    console.print(
        Panel(
            texto,
            title="[bold white]BLOCO OPERATÓRIO — SISTEMA DE AGENTES CIRÚRGICOS[/bold white]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def mostrar_tabela_documentos(rag: BaseConhecimentoRAG) -> None:
    """Exibe uma tabela formatada com todos os procedimentos cirúrgicos em `docs/`."""
    if not rag.documentos:
        console.print("[yellow]Não foram encontrados documentos na pasta `docs/`.[/yellow]")
        return

    tabela = Table(
        title="Documentos Cirúrgicos Indexados em `docs/`",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
    )
    tabela.add_column("#", style="bold white", width=4)
    tabela.add_column("Código", style="cyan", width=9)
    tabela.add_column("Procedimento Cirúrgico", style="bold white")
    tabela.add_column("Especialidade", style="green")
    tabela.add_column("Duração", style="yellow")
    tabela.add_column("Ficheiro", style="dim")

    for idx, doc in enumerate(rag.documentos, start=1):
        tabela.add_row(
            str(idx),
            doc.codigo,
            doc.titulo,
            doc.especialidade,
            doc.duracao,
            doc.ficheiro,
        )

    console.print(tabela)


def mostrar_relatorios_guardados(pasta_relatorios: Path) -> None:
    """Lista os relatórios Markdown já gerados na pasta `relatorios/`."""
    if not pasta_relatorios.exists():
        console.print("[yellow]Ainda não existem relatórios guardados em `relatorios/`.[/yellow]")
        return

    ficheiros = sorted(pasta_relatorios.glob("*.md"), reverse=True)
    if not ficheiros:
        console.print("[yellow]Ainda não existem relatórios guardados em `relatorios/`.[/yellow]")
        return

    tabela = Table(
        title="Relatórios Gerados em `relatorios/`",
        header_style="bold magenta",
        border_style="bright_black",
    )
    tabela.add_column("#", width=4)
    tabela.add_column("Ficheiro", style="bold white")
    tabela.add_column("Tamanho", style="cyan", justify="right")

    for idx, f in enumerate(ficheiros[:15], start=1):
        tamanho_kb = f.stat().st_size / 1024
        tabela.add_row(str(idx), f.name, f"{tamanho_kb:.1f} KB")

    console.print(tabela)


def mostrar_documento_original(rag: BaseConhecimentoRAG, termo: str) -> None:
    """Mostra o conteúdo original de um documento da pasta `docs/`."""
    termo_limpo = termo.strip().lower()
    escolhido = None

    if termo_limpo.isdigit():
        idx = int(termo_limpo) - 1
        if 0 <= idx < len(rag.documentos):
            escolhido = rag.documentos[idx]
    else:
        for doc in rag.documentos:
            if (
                termo_limpo in doc.codigo.lower()
                or termo_limpo in doc.ficheiro.lower()
                or termo_limpo in normalizar_texto(doc.titulo)
            ):
                escolhido = doc
                break

    if not escolhido:
        console.print(
            f"[red]Documento '{termo}' não encontrado. Use `/docs` para ver os números e códigos.[/red]"
        )
        return

    console.print(
        Panel(
            Markdown(escolhido.conteudo_completo),
            title=f"[bold cyan]Documento Fonte: docs/{escolhido.ficheiro}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def mostrar_ajuda() -> None:
    """Mostra os comandos disponíveis e sugestões de pesquisas."""
    texto_ajuda = """
### Comandos Disponíveis no Terminal
- **`/docs`** — Lista todos os documentos cirúrgicos disponíveis na pasta `docs/`.
- **`/ver <n>`** — Abre e mostra o documento original n.º `<n>` (ex: `/ver 1` ou `/ver CG-002`).
- **`/relatorios`** — Lista os relatórios gerados e guardados na pasta `relatorios/`.
- **`/recarga`** — Recarrega os ficheiros da pasta `docs/` (útil se adicionares novos documentos).
- **`/ajuda`** — Mostra este painel de ajuda.
- **`/sair`** — Sai da aplicação.

### Exemplos de Pesquisas para os Agentes
1. *"Quais os passos e instrumentos necessários para realizar uma colecistectomia laparoscópica?"*
2. *"Prepara um relatório completo sobre a artroplastia total da anca e os cuidados com o nervo ciático."*
3. *"Que instrumentos de microcirurgia e etapas são usados no bypass coronário (CABG)?"*
4. *"Como é feita a hernioplastia inguinal de Lichtenstein e que fios de sutura são precisos?"*
5. *"Quais os instrumentos e etapas da cesariana segmentar transversal?"*
6. *"Protocolo cirúrgico e instrumentos para drenagem de hematoma subdural agudo por craniotomia."*
"""
    console.print(
        Panel(
            Markdown(texto_ajuda.strip()),
            title="[bold green]Ajuda e Exemplos de Pesquisa[/bold green]",
            border_style="green",
        )
    )


def executar_pesquisa_com_interface(
    equipa: EquipaAgentesCirurgicos,
    pesquisa: str,
    pasta_relatorios: Path,
    mostrar_etapas_intermedias: bool = False,
) -> Optional[RelatorioCirurgico]:
    """Executa o fluxo de agentes mostrando o estado em tempo real no terminal."""
    console.print()
    console.print(Rule(f"[bold cyan]Nova Pesquisa Cirúrgica[/bold cyan]"))

    # Mostrar documentos selecionados pelo motor RAG antes de invocar os agentes
    res_rag = equipa.rag.pesquisar(pesquisa)
    if not res_rag.documentos_relevantes:
        console.print("[red]Não existem documentos em `docs/` para responder à pesquisa.[/red]")
        return None

    tabela_rag = Table(
        title="Documentos Selecionados pelo Motor RAG (`docs/`)",
        header_style="bold green",
        border_style="green",
    )
    tabela_rag.add_column("Código", style="cyan")
    tabela_rag.add_column("Procedimento", style="bold white")
    tabela_rag.add_column("Ficheiro Fonte", style="dim")
    tabela_rag.add_column("Score RAG", justify="right", style="bold green")

    for doc, score in res_rag.documentos_relevantes:
        tabela_rag.add_row(doc.codigo, doc.titulo, f"docs/{doc.ficheiro}", f"{score:.2f}")

    console.print(tabela_rag)
    console.print()

    status_spinner: Optional[Status] = None

    def ao_iniciar_agente(passo: int, total: int, nome_agente: str, etapa: str) -> None:
        nonlocal status_spinner
        if status_spinner is not None:
            status_spinner.stop()
        status_spinner = console.status(
            f"[bold cyan][{passo}/{total}] {nome_agente}[/bold cyan] a iniciar ({etapa})...",
            spinner="dots",
        )
        status_spinner.start()

    def ao_receber_token(passo: int, total: int, nome_agente: str, _tok: str, n_tokens: int) -> None:
        nonlocal status_spinner
        if status_spinner is not None and (n_tokens == 1 or n_tokens % 10 == 0):
            status_spinner.update(
                f"[bold cyan][{passo}/{total}] {nome_agente}[/bold cyan] a gerar resposta... "
                f"[bold yellow]({n_tokens} tokens recebidos)[/bold yellow]"
            )

    def ao_concluir_agente(registo: RegistoEtapaAgente) -> None:
        nonlocal status_spinner
        if status_spinner is not None:
            status_spinner.stop()
            status_spinner = None

        console.print(
            f"  [bold green]✓[/bold green] [bold white]{registo.nome_agente}[/bold white] "
            f"concluído em [cyan]{registo.tempo_segundos:.1f}s[/cyan] "
            f"([yellow]{registo.tokens_gerados} tokens[/yellow]) "
            f"[dim]— {registo.etapa_assegurada}[/dim]"
        )

        if mostrar_etapas_intermedias:
            console.print(
                Panel(
                    Markdown(registo.conteudo),
                    title=f"[bold magenta]{registo.nome_agente}[/bold magenta]",
                    subtitle=f"[dim]{registo.papel}[/dim]",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

    try:
        relatorio = equipa.processar_pesquisa(
            pesquisa_utilizador=pesquisa,
            callback_progresso=ao_iniciar_agente,
            callback_token_agente=ao_receber_token,
            callback_conclusao_agente=ao_concluir_agente,
        )
    except ErroConexaoOllama as exc:
        if status_spinner is not None:
            status_spinner.stop()
        console.print(
            Panel(
                f"[bold red]Falha na comunicação com o Ollama:[/bold red]\n{exc}\n\n"
                "[yellow]Dica:[/yellow] Verifique se o serviço está ativo com `ollama serve` "
                f"e se o modelo `{equipa.cliente.modelo}` está instalado (`ollama list`).",
                title="[bold red]Erro de Execução[/bold red]",
                border_style="red",
            )
        )
        return None
    except KeyboardInterrupt:
        if status_spinner is not None:
            status_spinner.stop()
        console.print("\n[yellow]Geração interrompida pelo utilizador.[/yellow]")
        return None

    caminho_guardado = guardar_relatorio(relatorio, pasta_relatorios)

    console.print()
    console.print(
        Panel(
            Markdown(relatorio.markdown_final),
            title="[bold green]RELATÓRIO TÉCNICO CIRÚRGICO FINAL[/bold green]",
            subtitle=f"[bold cyan]Guardado em: {caminho_guardado}[/bold cyan]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print(
        f"[bold green]Relatório exportado com sucesso para:[/bold green] [underline]{caminho_guardado}[/underline]\n"
    )
    return relatorio


def parse_args() -> argparse.Namespace:
    """Configura os argumentos de linha de comandos."""
    parser = argparse.ArgumentParser(
        description="App de Terminal Multi-Agente para Relatórios de Procedimentos Cirúrgicos (Ollama + RAG)."
    )
    parser.add_argument(
        "pesquisa",
        nargs="?",
        default=None,
        help="Pesquisa cirúrgica a realizar diretamente (se omitida, inicia o modo interativo).",
    )
    parser.add_argument(
        "--modelo",
        default=MODELO_PADRAO,
        help=f"Nome do modelo no Ollama (predefinido: {MODELO_PADRAO}).",
    )
    parser.add_argument(
        "--url",
        default=OLLAMA_URL_PADRAO,
        help=f"URL base do servidor Ollama (predefinido: {OLLAMA_URL_PADRAO}).",
    )
    parser.add_argument(
        "--docs",
        default="docs",
        help="Caminho para a pasta de documentos cirúrgicos (predefinido: docs).",
    )
    parser.add_argument(
        "--output",
        default="relatorios",
        help="Pasta onde os relatórios Markdown são guardados (predefinido: relatorios).",
    )
    parser.add_argument(
        "--mostrar-etapas",
        action="store_true",
        help="Exibe o painel individual de cada agente à medida que termina.",
    )
    parser.add_argument(
        "--listar-docs",
        action="store_true",
        help="Lista os documentos cirúrgicos indexados em docs/ e termina.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pasta_base = Path(__file__).resolve().parent
    pasta_docs = Path(args.docs) if Path(args.docs).is_absolute() else pasta_base / args.docs
    pasta_relatorios = (
        Path(args.output) if Path(args.output).is_absolute() else pasta_base / args.output
    )

    rag = BaseConhecimentoRAG(pasta_docs=pasta_docs)
    cliente = ClienteOllama(modelo=args.modelo, base_url=args.url)
    equipa = EquipaAgentesCirurgicos(rag=rag, cliente=cliente)

    if args.listar_docs:
        mostrar_tabela_documentos(rag)
        return 0

    estado_ok, msg_estado, _ = cliente.verificar_estado()
    mostrar_cabecalho(rag, cliente, estado_ok, msg_estado)

    # Modo não-interativo (pesquisa passada como argumento CLI)
    if args.pesquisa:
        rel = executar_pesquisa_com_interface(
            equipa=equipa,
            pesquisa=args.pesquisa,
            pasta_relatorios=pasta_relatorios,
            mostrar_etapas_intermedias=args.mostrar_etapas,
        )
        return 0 if rel is not None else 1

    # Modo interativo no terminal
    mostrar_tabela_documentos(rag)
    console.print(
        "[dim]Escreva a sua pesquisa sobre um procedimento cirúrgico, ou digite "
        "[bold cyan]/ajuda[/bold cyan], [bold cyan]/docs[/bold cyan], "
        "[bold cyan]/ver <n>[/bold cyan], [bold cyan]/relatorios[/bold cyan] ou "
        "[bold cyan]/sair[/bold cyan].[/dim]\n"
    )

    while True:
        try:
            entrada = Prompt.ask("[bold cyan]Pesquisa Cirúrgica[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Sessão terminada. Até breve![/bold cyan]")
            break

        if not entrada:
            continue

        cmd = entrada.lower()
        if cmd in ("/sair", "sair", "exit", "quit", "q"):
            console.print("[bold cyan]Sessão terminada. Até breve![/bold cyan]")
            break
        elif cmd in ("/ajuda", "ajuda", "help", "/help"):
            mostrar_ajuda()
        elif cmd in ("/docs", "docs"):
            mostrar_tabela_documentos(rag)
        elif cmd in ("/relatorios", "relatorios"):
            mostrar_relatorios_guardados(pasta_relatorios)
        elif cmd in ("/recarga", "recarga"):
            total = rag.carregar_documentos()
            console.print(f"[green]Base documental recarregada: {total} documentos indexados.[/green]")
            mostrar_tabela_documentos(rag)
        elif cmd.startswith("/ver "):
            termo_ver = entrada[5:].strip()
            mostrar_documento_original(rag, termo_ver)
        else:
            executar_pesquisa_com_interface(
                equipa=equipa,
                pesquisa=entrada,
                pasta_relatorios=pasta_relatorios,
                mostrar_etapas_intermedias=args.mostrar_etapas,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())

