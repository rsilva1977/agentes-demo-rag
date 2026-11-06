# Sistema Multi-Agente de Relatórios Cirúrgicos (RAG + Ollama)

Aplicação de terminal em Python que utiliza uma equipa de **4 agentes especializados** alimentados pelo modelo local **`ministral-3:3b`** (via **Ollama**) e por um motor de **RAG (Retrieval-Augmented Generation)** sobre documentos técnicos cirúrgicos na pasta `docs/`.

---

## Arquitetura de Agentes

Cada agente assegura uma etapa distinta do procedimento cirúrgico com base exclusiva nos protocolos registados em `docs/`:

1. **Agente 1 — Triagem e Enquadramento Clínico (`AgentePesquisador`)**
   - Analisa a pesquisa do utilizador e os documentos recuperados pelo motor RAG.
   - Identifica o procedimento, código, especialidade, duração estimada, tipo de anestesia e indicações clínicas.
2. **Agente 2 — Preparação e Instrumentação Cirúrgica (`AgenteInstrumentacao`)**
   - Assegura a etapa pré-operatória, profilaxia antibiótica, posicionamento cirúrgico do doente e da equipa.
   - Organiza a mesa operatória listando todos os instrumentos cirúrgicos, equipamentos, afastadores, trocartes/motores e suturas por categoria funcional.
3. **Agente 3 — Condução Técnica e Etapas Operatórias (`AgenteProcedimento`)**
   - Assegura a execução cronológica passo a passo de todas as etapas da operação (acesso/incisão, exposição, tempo principal, hemostase e encerramento).
   - Destaca os cuidados críticos de segurança intraoperatória e estruturas anatómicas a preservar.
4. **Agente 4 — Auditoria e Síntese Executiva (`AgenteRelator`)**
   - Consolida as análises dos agentes anteriores, responde diretamente à questão do utilizador e gera uma **Checklist de Verificação Operatória** final.

---

## Documentos Cirúrgicos Incluídos (`docs/`)

| Código | Procedimento Cirúrgico | Especialidade | Ficheiro |
| :--- | :--- | :--- | :--- |
| `CG-001` | Apendicectomia Laparoscópica | Cirurgia Geral | `docs/01_apendicectomia_laparoscopica.md` |
| `CG-002` | Colecistectomia Laparoscópica | Cirurgia Geral / Hepatobiliar | `docs/02_colecistectomia_laparoscopica.md` |
| `ORT-001` | Artroplastia Total da Anca (Via Posterolateral) | Ortopedia e Traumatologia | `docs/03_artroplastia_total_anca.md` |
| `CG-003` | Hernioplastia Inguinal (Técnica de Lichtenstein) | Cirurgia Geral (Parede Abdominal) | `docs/04_hernioplastia_inguinal_lichtenstein.md` |
| `NC-001` | Craniotomia Descompressiva e Drenagem de Hematoma Subdural Agudo | Neurocirurgia | `docs/05_craniotomia_hematoma_subdural.md` |
| `CCT-001` | Cirurgia de Revascularização do Miocárdio (Bypass Coronário / CABG) | Cirurgia Cardiotorácica | `docs/06_revascularizacao_miocardio_cabg.md` |
| `OB-001` | Cesariana Segmentar Transversal (Técnica de Pfannenstiel / Kerr) | Ginecologia e Obstetrícia | `docs/07_cesariana_segmentar_transversal.md` |

> Podes adicionar novos ficheiros `.md` ou `.txt` na pasta `docs/` a qualquer momento (e usar o comando `/recarga` na aplicação).

---

## Pré-requisitos e Utilização

1. Certifica-te de que o **Ollama** está a correr localmente com o modelo `ministral-3:3b`:
   ```bash
   ollama pull ministral-3:3b
   ollama serve
   ```

2. Executa a aplicação em **modo interativo**:
   ```bash
   python3 main.py
   ```

3. Ou executa uma **pesquisa direta** via linha de comandos:
   ```bash
   python3 main.py "Quais os passos e instrumentos necessários para uma colecistectomia laparoscópica?"
   ```

4. Para visualizar também o painel individual de cada agente à medida que termina:
   ```bash
   python3 main.py --mostrar-etapas "Instrumentos e etapas da artroplastia total da anca"
   ```

5. Todos os relatórios gerados são apresentados no terminal e guardados automaticamente em formato Markdown na pasta `relatorios/`.

