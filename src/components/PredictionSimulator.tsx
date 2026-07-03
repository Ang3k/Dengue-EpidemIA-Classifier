import { useState } from "react";
import {
  formatModelName,
  solicitarSimulacaoRandom,
} from "../services/dengueRules";

type Pessoa = {
  idade: number | null;
  sexo: string | null;
  raca: string | null;
  ocupacao: string | null;
  estado: string | null;
  municipio: string | null;
  sintomas: string[];
};

type Predicao = {
  modelos: { name: string; probability: number; weight: number }[];
  media: number;
  threshold: number;
  ehDengue: boolean;
};

function PredictionSimulator() {
  const [pessoa, setPessoa] = useState<Pessoa | null>(null);
  const [predicao, setPredicao] = useState<Predicao | null>(null);
  const [classificacaoObservada, setClassificacaoObservada] =
    useState<string | null>(null);
  const [carregando, setCarregando] = useState(false);
  const [erro, setErro] = useState<string | null>(null);

  async function handleGerar() {
    setCarregando(true);
    setErro(null);
    setPessoa(null);
    setPredicao(null);
    setClassificacaoObservada(null);

    try {
      const resultado = await solicitarSimulacaoRandom();
      setPessoa({
        idade: resultado.case.age,
        sexo: resultado.case.sex,
        raca: resultado.case.race,
        ocupacao: resultado.case.occupation,
        estado: resultado.case.state,
        municipio: resultado.case.municipality,
        sintomas: resultado.case.symptoms,
      });
      setClassificacaoObservada(resultado.observedClassification);
      setPredicao({
        modelos: resultado.prediction.models,
        media: resultado.prediction.average,
        threshold: resultado.prediction.threshold,
        ehDengue: resultado.prediction.isDengue,
      });
    } catch (error) {
      setErro(
        error instanceof Error
          ? error.message
          : "Não foi possível carregar a simulação histórica."
      );
    } finally {
      setCarregando(false);
    }
  }

  const realidadeDengue = (classificacaoObservada ?? "")
    .toLowerCase()
    .includes("dengue");

  return (
    <div className="home-section">
      <h2>Simulação de predição</h2>
      <p>
        Gere um caso histórico real do conjunto de teste e veja a predição dos
        modelos treinados. O backend seleciona o caso, aplica o mesmo
        pré-processamento do treino e retorna o resultado completo.
      </p>

      {erro && <p style={{ color: "#dc2626", fontWeight: 600 }}>{erro}</p>}

      {pessoa && predicao && (
        <div className="sim-card">
          {/* Topo: dados da pessoa */}
          <div className="sim-dados">
            <div className="sim-campo">
              <span className="sim-label">Idade</span>
              <span className="sim-valor">
                {pessoa.idade !== null ? `${pessoa.idade} anos` : "Não informado"}
              </span>
            </div>
            <div className="sim-campo">
              <span className="sim-label">Sexo</span>
              <span className="sim-valor">{pessoa.sexo ?? "Não informado"}</span>
            </div>
            <div className="sim-campo">
              <span className="sim-label">Raça/cor</span>
              <span className="sim-valor">{pessoa.raca ?? "Não informado"}</span>
            </div>
            <div className="sim-campo">
              <span className="sim-label">UF</span>
              <span className="sim-valor">{pessoa.estado ?? "Não informado"}</span>
            </div>
            <div className="sim-campo sim-campo-largo">
              <span className="sim-label">Ocupação</span>
              <span className="sim-valor">{pessoa.ocupacao ?? "Não informado"}</span>
            </div>
            <div className="sim-campo sim-campo-largo">
              <span className="sim-label">Município</span>
              <span className="sim-valor">
                {pessoa.municipio ?? "Não informado"}
              </span>
            </div>
          </div>

          <div className="sim-sintomas">
            <span className="sim-label">Sintomas informados</span>
            <div className="sim-tags">
              {pessoa.sintomas.map((sintoma) => (
                <span key={sintoma} className="sim-tag">
                  {sintoma}
                </span>
              ))}
            </div>
          </div>

          {/* Meio: bloquinhos dos modelos */}
          <span className="sim-label">Resultado dos modelos</span>
          <div className="modelo-quadrados">
            {predicao.modelos.map((modelo) => (
              <div className="modelo-quadrado" key={modelo.name}>
                <span className="modelo-quadrado-nome">
                  {formatModelName(modelo.name)}
                </span>
                <span className="modelo-quadrado-prob">
                  {modelo.probability}%
                </span>
                <span className="modelo-quadrado-legenda">prob. de dengue</span>
                <span className="modelo-quadrado-peso">
                  Peso no ensemble: {modelo.weight}%
                </span>
              </div>
            ))}
          </div>

          {/* Baixo: previsão final (esquerda) x realidade (direita) */}
          <div className="sim-final-grid">
            <div
              className={`sim-veredito ${
                predicao.ehDengue ? "sim-veredito-dengue" : "sim-veredito-nao"
              }`}
            >
              <span className="sim-veredito-titulo">
                Previsão do modelo (ensemble)
              </span>
              {predicao.ehDengue ? "É dengue" : "Não é dengue"}
              <small>
                Score {predicao.media}%, limiar de {predicao.threshold}%
              </small>
            </div>

            <div
              className={`sim-veredito ${
                realidadeDengue ? "sim-veredito-dengue" : "sim-veredito-nao"
              }`}
            >
              <span className="sim-veredito-titulo">
                Realidade (base histórica)
              </span>
              {classificacaoObservada ?? "Não informado"}
              <small>Diagnóstico registrado no SINAN</small>
            </div>
          </div>
        </div>
      )}

      {/* Botão sempre no final, para gerar uma nova sem rolar para cima */}
      <button
        type="button"
        className="btn-primary"
        onClick={handleGerar}
        disabled={carregando}
      >
        {carregando ? "Carregando caso histórico..." : "Gerar simulação real"}
      </button>
    </div>
  );
}

export default PredictionSimulator;
