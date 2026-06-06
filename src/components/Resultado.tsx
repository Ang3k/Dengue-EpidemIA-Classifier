type ModelResult = {
  name: string;
  result: string;
  confidence: string;
  description: string;
};

type ResultadoProps = {
  title: string;
  message: string;
  level: string;
  models: {
    clinical: ModelResult;
    epidemiological: ModelResult;
  };
};

function Resultado({ title, message, level, models }: ResultadoProps) {
  return (
    <div className={`resultado resultado-${level.toLowerCase()}`}>
      <h2>{title}</h2>

      <p>{message}</p>

      <div className="modelos-container">
        <div className="modelo-card">
          <div className="modelo-header">
            <span className="modelo-icon">🩺</span>
            <div>
              <h3>{models.clinical.name}</h3>
              <strong>{models.clinical.result}</strong>
            </div>
          </div>

          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{ width: models.clinical.confidence }}
            ></div>
          </div>

          <p className="confidence-text">
            Confiança simulada: {models.clinical.confidence}
          </p>

          <p>{models.clinical.description}</p>
        </div>

        <div className="modelo-card">
          <div className="modelo-header">
            <span className="modelo-icon">📊</span>
            <div>
              <h3>{models.epidemiological.name}</h3>
              <strong>{models.epidemiological.result}</strong>
            </div>
          </div>

          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{ width: models.epidemiological.confidence }}
            ></div>
          </div>

          <p className="confidence-text">
            Confiança simulada: {models.epidemiological.confidence}
          </p>

          <p>{models.epidemiological.description}</p>
        </div>
      </div>

      <div className="conclusao-modelos">
        <strong>Conclusão combinada:</strong>
        <p>
          A avaliação acima é uma simulação visual de dois modelos. Ela não
          substitui avaliação médica, exames laboratoriais ou atendimento em
          unidade de saúde.
        </p>
      </div>
    </div>
  );
}

export default Resultado;