import {
  DENGUE_THRESHOLD,
  formatModelName,
} from "../services/dengueRules";
import type { EvaluationResult } from "../services/dengueRules";

function Resultado({ models, average, isDengue }: EvaluationResult) {
  return (
    <div className="resultado-triagem">
      <h2>Resultado da triagem</h2>

      <span className="sim-label">Resultado dos modelos</span>
      <div className="modelo-quadrados">
        {models.map((modelo) => (
          <div className="modelo-quadrado" key={modelo.name}>
            <span className="modelo-quadrado-nome">
              {formatModelName(modelo.name)}
            </span>
            <span className="modelo-quadrado-prob">{modelo.probability}%</span>
            <span className="modelo-quadrado-legenda">prob. de dengue</span>
          </div>
        ))}
      </div>

      <div className="predicao-media">
        <span className="sim-label">Probabilidade média</span>
        <span className="sim-valor-destaque">{average}%</span>
      </div>

      <div
        className={`sim-veredito ${
          isDengue ? "sim-veredito-dengue" : "sim-veredito-nao"
        }`}
      >
        {isDengue ? "É dengue" : "Não é dengue"}
        <small>
          Média {isDengue ? "acima" : "abaixo"} do limiar de {DENGUE_THRESHOLD}%
        </small>
      </div>

      <div className="orientacao-final">
        <strong>Orientação:</strong>
        <p>
          Esta triagem é apenas informativa e não substitui avaliação médica.
          Em caso de piora, febre persistente, sangramentos, dor abdominal
          intensa, vômitos persistentes ou sonolência, procure atendimento em
          uma unidade de saúde.
        </p>
      </div>
    </div>
  );
}

export default Resultado;
