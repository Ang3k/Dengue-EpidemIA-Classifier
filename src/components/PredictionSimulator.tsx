import { useState } from "react";
import { DENGUE_THRESHOLD } from "../services/dengueRules";

const API_URL = "http://localhost:8000";

// Valores baseados nos mapeamentos do dengue_pipeline
const SEXOS = [
  { label: "Masculino", code: "M" },
  { label: "Feminino", code: "F" },
];

const RACAS = [
  { label: "Branca", code: 1 },
  { label: "Preta", code: 2 },
  { label: "Amarela", code: 3 },
  { label: "Parda", code: 4 },
  { label: "Indígena", code: 5 },
];

const ESCOLARIDADES = [
  { label: "Analfabeto", code: 1 },
  { label: "Ensino fundamental completo", code: 4 },
  { label: "Ensino médio incompleto", code: 5 },
  { label: "Ensino médio completo", code: 6 },
  { label: "Educação superior incompleta", code: 7 },
  { label: "Educação superior completa", code: 8 },
];

const OCUPACOES = [
  "Estudante",
  "Dona de casa",
  "Trabalhador agropecuário em geral",
  "Pedreiro",
  "Motorista de carro de passeio",
  "Vendedor de comércio varejista",
  "Professor de nível médio no ensino fundamental",
  "Técnico de enfermagem",
  "Auxiliar de escritório, em geral",
  "Recepcionista, em geral",
  "Empregado doméstico nos serviços gerais",
  "Cozinheiro geral",
  "Vigilante",
  "Operador de caixa",
];

// Sintomas com seus ids para a API
const SINTOMAS = [
  { label: "Febre", id: "fever" },
  { label: "Mialgia / dor muscular", id: "myalgia" },
  { label: "Cefaleia / dor de cabeça", id: "headache" },
  { label: "Exantema / manchas na pele", id: "rash" },
  { label: "Vômitos", id: "vomiting" },
  { label: "Náusea / enjoo", id: "nausea" },
  { label: "Dor nas costas", id: "back_pain" },
  { label: "Conjuntivite", id: "conjunctivitis" },
  { label: "Dor nas articulações", id: "joint_pain" },
  { label: "Dor atrás dos olhos", id: "retro_orbital_pain" },
];

const CLASSIFICACOES = [
  "Descartado",
  "Dengue",
  "Dengue com sinais de alarme",
  "Dengue grave",
];

// UFs do Brasil com código IBGE
const UFS = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53];

type SintomaItem = { label: string; id: string };

type Pessoa = {
  idade: number;
  sexo: { label: string; code: string };
  raca: { label: string; code: number };
  escolaridade: { label: string; code: number };
  ocupacao: string;
  sintomas: SintomaItem[];
  residenceState: number;
  notificationMonth: number;
  symptomEpiWeekNumber: number;
  classificacaoReal: string;
};

type Predicao = {
  modelos: { name: string; probability: number }[];
  media: number;
  ehDengue: boolean;
};

function escolherAleatorio<T>(lista: T[]): T {
  return lista[Math.floor(Math.random() * lista.length)];
}

function gerarPessoa(): Pessoa {
  const sintomasEmbaralhados = [...SINTOMAS].sort(() => Math.random() - 0.5);
  const quantidade = 2 + Math.floor(Math.random() * 4);

  return {
    idade: 1 + Math.floor(Math.random() * 89),
    sexo: escolherAleatorio(SEXOS),
    raca: escolherAleatorio(RACAS),
    escolaridade: escolherAleatorio(ESCOLARIDADES),
    ocupacao: escolherAleatorio(OCUPACOES),
    sintomas: sintomasEmbaralhados.slice(0, quantidade),
    residenceState: escolherAleatorio(UFS),
    notificationMonth: 1 + Math.floor(Math.random() * 12),
    symptomEpiWeekNumber: 1 + Math.floor(Math.random() * 52),
    classificacaoReal: escolherAleatorio(CLASSIFICACOES),
  };
}

async function chamarAPI(pessoa: Pessoa): Promise<Predicao> {
  const sintomasPayload: Record<string, number> = {};
  for (const s of SINTOMAS) {
    sintomasPayload[s.id] = pessoa.sintomas.some((ps) => ps.id === s.id) ? 1 : 0;
  }

  const payload = {
    age_years: pessoa.idade,
    sex: pessoa.sexo.code,
    race: pessoa.raca.code,
    education_level: pessoa.escolaridade.code,
    occupation_code: "1",
    residence_state: pessoa.residenceState,
    notification_month: pessoa.notificationMonth,
    symptom_epi_week_number: pessoa.symptomEpiWeekNumber,
    ...sintomasPayload,
  };

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) throw new Error(`Erro na API: ${response.status}`);

  const data = await response.json();
  return {
    modelos: data.models,
    media: data.average,
    ehDengue: data.isDengue,
  };
}

function PredictionSimulator() {
  const [pessoa, setPessoa] = useState<Pessoa | null>(null);
  const [predicao, setPredicao] = useState<Predicao | null>(null);
  const [carregando, setCarregando] = useState(false);
  const [erro, setErro] = useState<string | null>(null);

  function handleGerar() {
    setPessoa(gerarPessoa());
    setPredicao(null);
    setErro(null);
  }

  async function handleRodarPredicao() {
    if (!pessoa) return;
    setCarregando(true);
    setErro(null);
    setPredicao(null);

    try {
      const resultado = await chamarAPI(pessoa);
      setPredicao(resultado);
    } catch {
      setErro("Não foi possível conectar à API. Verifique se o servidor está rodando.");
    } finally {
      setCarregando(false);
    }
  }

  return (
    <div className="home-section">
      <h2>Simulação de predição</h2>
      <p>
        Gere uma pessoa com dados aleatórios e rode a predição para ver como o
        sistema funciona: os modelos treinados avaliam o caso, cada um com sua
        probabilidade, e a média define o resultado final.
      </p>

      <button type="button" className="btn-primary" onClick={handleGerar}>
        Gerar pessoa aleatória
      </button>

      {pessoa && (
        <div className="sim-card">
          <div className="sim-dados">
            <div className="sim-campo">
              <span className="sim-label">Idade</span>
              <span className="sim-valor">{pessoa.idade} anos</span>
            </div>
            <div className="sim-campo">
              <span className="sim-label">Sexo</span>
              <span className="sim-valor">{pessoa.sexo.label}</span>
            </div>
            <div className="sim-campo">
              <span className="sim-label">Raça/cor</span>
              <span className="sim-valor">{pessoa.raca.label}</span>
            </div>
            <div className="sim-campo">
              <span className="sim-label">Escolaridade</span>
              <span className="sim-valor">{pessoa.escolaridade.label}</span>
            </div>
            <div className="sim-campo sim-campo-largo">
              <span className="sim-label">Ocupação</span>
              <span className="sim-valor">{pessoa.ocupacao}</span>
            </div>
          </div>

          <div className="sim-sintomas">
            <span className="sim-label">Sintomas informados</span>
            <div className="sim-tags">
              {pessoa.sintomas.map((sintoma) => (
                <span key={sintoma.id} className="sim-tag">
                  {sintoma.label}
                </span>
              ))}
            </div>
          </div>

          <div className="sim-classificacao">
            <span className="sim-label">Classificação real</span>
            <span className="sim-valor-destaque">{pessoa.classificacaoReal}</span>
          </div>

          <button
            type="button"
            className="btn-predicao"
            onClick={handleRodarPredicao}
            disabled={carregando}
          >
            {carregando ? "Calculando..." : "Rodar predição"}
          </button>

          {erro && (
            <p style={{ color: "red", marginTop: "1rem" }}>{erro}</p>
          )}

          {predicao && (
            <div className="sim-predicao">
              <span className="sim-label">Resultado dos modelos</span>

              <div className="sim-modelos">
                {predicao.modelos.map((modelo) => (
                  <div key={modelo.name} className="sim-modelo">
                    <div className="sim-modelo-topo">
                      <span className="sim-modelo-nome">{modelo.name}</span>
                      <span className="sim-modelo-prob">
                        {modelo.probability}%
                      </span>
                    </div>
                    <div className="sim-barra">
                      <div
                        className="sim-barra-preench"
                        style={{ width: `${modelo.probability}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="sim-media">
                <span className="sim-label">Probabilidade média</span>
                <span className="sim-valor-destaque">{predicao.media}%</span>
              </div>

              <div
                className={`sim-veredito ${
                  predicao.ehDengue ? "sim-veredito-dengue" : "sim-veredito-nao"
                }`}
              >
                {predicao.ehDengue ? "É dengue" : "Não é dengue"}
                <small>
                  Média {predicao.ehDengue ? "acima" : "abaixo"} do limiar de{" "}
                  {DENGUE_THRESHOLD}%
                </small>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default PredictionSimulator;