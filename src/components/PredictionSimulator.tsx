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

  return (
    <div className="home-section">
      <h2>Simulação de predição</h2>
      <p>
        Gere um caso histórico real do conjunto de teste e veja a predição dos
        modelos treinados. O backend seleciona o caso, aplica o mesmo
        pré-processamento do treino e retorna o resultado completo.
      </p>

      <button type="button" className="btn-primary" onClick={handleGerar}>
        {carregando ? "Carregando caso histórico..." : "Gerar simulação real"}
      </button>

      {pessoa && (
        <div className="sim-card">
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
              <span className="sim-valor">{pessoa.municipio ?? "Não informado"}</span>
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

          {classificacaoObservada && (
            <div className="sim-media sim-media-observada">
              <span className="sim-label">Classificação observada</span>
              <span className="sim-valor-destaque">{classificacaoObservada}</span>
              <small className="sim-media-apoio">
                Este é o diagnóstico registrado na base histórica. Pode divergir
                da predição dos modelos.
              </small>
            </div>
          )}

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
                        <small className="sim-modelo-peso">
                          Peso: {modelo.weight}%
                        </small>
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
                <span className="sim-label">Score ponderado por recall</span>
                <span className="sim-valor-destaque">{predicao.media}%</span>
              </div>

              <div
                className={`sim-veredito ${
                  predicao.ehDengue ? "sim-veredito-dengue" : "sim-veredito-nao"
                }`}
              >
                {predicao.ehDengue ? "É dengue" : "Não é dengue"}
                <small>
                  Score {predicao.ehDengue ? "acima" : "abaixo"} do limiar de{" "}
                  {predicao.threshold}%
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
