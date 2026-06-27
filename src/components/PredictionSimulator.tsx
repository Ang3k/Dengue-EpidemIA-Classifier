import { useState } from "react";
import {
  DENGUE_THRESHOLD,
  formatModelName,
  solicitarPredicao,
  triageItems,
} from "../services/dengueRules";
import type { PredictionPayload } from "../services/dengueRules";

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
  { label: "Estudante", code: "999991" },
  { label: "Dona de casa", code: "999992" },
  { label: "Trabalhador agropecuário em geral", code: "621005" },
  { label: "Pedreiro", code: "715210" },
  { label: "Motorista de carro de passeio", code: "782305" },
  { label: "Vendedor de comércio varejista", code: "521110" },
  {
    label: "Professor de nível médio no ensino fundamental",
    code: "331205",
  },
  { label: "Técnico de enfermagem", code: "322205" },
  { label: "Auxiliar de escritório, em geral", code: "411005" },
  { label: "Recepcionista, em geral", code: "422105" },
  {
    label: "Empregado doméstico nos serviços gerais",
    code: "512105",
  },
  { label: "Cozinheiro geral", code: "513205" },
  { label: "Vigilante", code: "517330" },
  { label: "Operador de caixa", code: "421125" },
];

const SINTOMAS = triageItems.map(({ id, label }) => ({ id, label }));

// UFs do Brasil com código IBGE
const UFS = [
  11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31,
  32, 33, 35, 41, 42, 43, 50, 51, 52, 53,
];

type SintomaItem = { label: string; id: string };

type Pessoa = {
  idade: number;
  sexo: { label: string; code: string };
  raca: { label: string; code: number };
  escolaridade: { label: string; code: number };
  ocupacao: { label: string; code: string };
  sintomas: SintomaItem[];
  residenceState: number;
  notificationDate: string;
  symptomOnsetDate: string;
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
  const inicioSintomas = new Date(
    Date.UTC(
      2019,
      Math.floor(Math.random() * 12),
      1 + Math.floor(Math.random() * 24)
    )
  );
  const notificacao = new Date(inicioSintomas);
  notificacao.setUTCDate(
    notificacao.getUTCDate() + Math.floor(Math.random() * 8)
  );

  return {
    idade: 1 + Math.floor(Math.random() * 89),
    sexo: escolherAleatorio(SEXOS),
    raca: escolherAleatorio(RACAS),
    escolaridade: escolherAleatorio(ESCOLARIDADES),
    ocupacao: escolherAleatorio(OCUPACOES),
    sintomas: sintomasEmbaralhados.slice(0, quantidade),
    residenceState: escolherAleatorio(UFS),
    notificationDate: notificacao.toISOString().slice(0, 10),
    symptomOnsetDate: inicioSintomas.toISOString().slice(0, 10),
  };
}

async function chamarAPI(pessoa: Pessoa): Promise<Predicao> {
  const sintomasPayload: Record<string, number> = {};
  for (const s of SINTOMAS) {
    sintomasPayload[s.id] = pessoa.sintomas.some((ps) => ps.id === s.id) ? 1 : 0;
  }

  const payload: PredictionPayload = {
    age_years: pessoa.idade,
    sex: pessoa.sexo.code,
    race: pessoa.raca.code,
    education_level: pessoa.escolaridade.code,
    occupation_code: pessoa.ocupacao.code,
    residence_state: pessoa.residenceState,
    notification_date: pessoa.notificationDate,
    symptom_onset_date: pessoa.symptomOnsetDate,
    ...sintomasPayload,
  };

  const data = await solicitarPredicao(payload);
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
    } catch (error) {
      setErro(
        error instanceof Error
          ? error.message
          : "Não foi possível concluir a predição."
      );
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
              <span className="sim-valor">{pessoa.ocupacao.label}</span>
            </div>
            <div className="sim-campo">
              <span className="sim-label">Início dos sintomas</span>
              <span className="sim-valor">{pessoa.symptomOnsetDate}</span>
            </div>
            <div className="sim-campo">
              <span className="sim-label">Notificação</span>
              <span className="sim-valor">{pessoa.notificationDate}</span>
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
                      <span className="sim-modelo-nome">
                        {formatModelName(modelo.name)}
                      </span>
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
