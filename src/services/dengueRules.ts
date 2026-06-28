import type { PatientData } from "../types/patient";

export type TriageItem = {
  id: string;
  label: string;
  points: number;
  group: "symptoms" | "clinical";
};

export type ModelPrediction = {
  name: string;
  probability: number;
};

export type EvaluationResult = {
  models: ModelPrediction[];
  average: number;
  isDengue: boolean;
};

export type PredictionPayload = Record<string, string | number | null>;

export type RandomSimulationRequest = {
  seed?: number;
};

export type RandomSimulationCase = {
  age: number | null;
  sex: string | null;
  race: string | null;
  occupation: string | null;
  state: string | null;
  municipality: string | null;
  symptoms: string[];
};

export type RandomSimulationResponse = {
  case: RandomSimulationCase;
  observedClassification: string | null;
  prediction: EvaluationResult;
};

// Acima deste valor (em %), o resultado final é considerado dengue
export const DENGUE_THRESHOLD = 40;

const API_URL = (
  import.meta.env.VITE_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "Regressão logística",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  decision_tree: "Árvore de decisão",
};

export function formatModelName(name: string): string {
  return MODEL_LABELS[name] ?? name;
}

export const triageItems: TriageItem[] = [
  {
    id: "fever",
    label: "Febre",
    points: 3,
    group: "symptoms",
  },
  {
    id: "myalgia",
    label: "Mialgia / dor muscular",
    points: 2,
    group: "symptoms",
  },
  {
    id: "headache",
    label: "Cefaleia / dor de cabeça",
    points: 2,
    group: "symptoms",
  },
  {
    id: "rash",
    label: "Exantema / manchas na pele",
    points: 2,
    group: "symptoms",
  },
  {
    id: "vomiting",
    label: "Vômitos",
    points: 2,
    group: "symptoms",
  },
  {
    id: "nausea",
    label: "Náusea / enjoo",
    points: 1,
    group: "symptoms",
  },
  {
    id: "back_pain",
    label: "Dor nas costas",
    points: 1,
    group: "symptoms",
  },
  {
    id: "conjunctivitis",
    label: "Conjuntivite",
    points: 1,
    group: "symptoms",
  },
  {
    id: "arthritis",
    label: "Artrite",
    points: 1,
    group: "symptoms",
  },
  {
    id: "joint_pain",
    label: "Dor nas articulações",
    points: 2,
    group: "symptoms",
  },
  {
    id: "petechiae",
    label: "Petéquias / pequenos pontos vermelhos na pele",
    points: 2,
    group: "symptoms",
  },
  {
    id: "retro_orbital_pain",
    label: "Dor atrás dos olhos",
    points: 2,
    group: "symptoms",
  },
  {
    id: "tourniquet_test",
    label: "Prova do laço positiva",
    points: 3,
    group: "clinical",
  },
];

function numberOrNull(value: string): number | null {
  if (!value) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function montarPayload(
  selectedIds: string[],
  patientData: PatientData
): PredictionPayload {
  const sintomas: PredictionPayload = {};
  for (const item of triageItems) {
    sintomas[item.id] = selectedIds.includes(item.id) ? 1 : 0;
  }

  return {
    age_years: numberOrNull(patientData.ageYears),
    sex: patientData.sex || null,
    pregnancy_status: numberOrNull(patientData.pregnancyStatus),
    race: numberOrNull(patientData.race),
    education_level: numberOrNull(patientData.educationLevel),
    occupation_code: patientData.occupationCode || null,
    residence_state: numberOrNull(patientData.residenceState),
    residence_municipality: numberOrNull(
      patientData.residenceMunicipality
    ),
    residence_health_region: numberOrNull(
      patientData.residenceHealthRegion
    ),
    notification_date: patientData.notificationDate || null,
    symptom_onset_date: patientData.symptomOnsetDate || null,
    days_to_notification: numberOrNull(patientData.daysToNotification),
    symptom_epi_week_number: numberOrNull(
      patientData.symptomEpiWeekNumber
    ),
    ...sintomas,
  };
}

export async function solicitarPredicao(
  payload: PredictionPayload
): Promise<EvaluationResult> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(
      "Não foi possível conectar à API. Verifique se o servidor está rodando."
    );
  }

  if (!response.ok) {
    let detail = `A API retornou o status ${response.status}.`;
    try {
      const errorBody = await response.json();
      if (typeof errorBody.detail === "string") {
        detail = errorBody.detail;
      }
    } catch {
      // A resposta não contém JSON; usa a mensagem baseada no status.
    }
    throw new Error(detail);
  }

  const data: unknown = await response.json();
  if (
    !data ||
    typeof data !== "object" ||
    !Array.isArray((data as EvaluationResult).models) ||
    typeof (data as EvaluationResult).average !== "number" ||
    typeof (data as EvaluationResult).isDengue !== "boolean"
  ) {
    throw new Error("A API retornou uma resposta inválida.");
  }

  return data as EvaluationResult;
}

export async function solicitarSimulacaoRandom(
  payload?: RandomSimulationRequest
): Promise<RandomSimulationResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/v1/simulations/random`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload ?? {}),
    });
  } catch {
    throw new Error(
      "Não foi possível conectar à API. Verifique se o servidor está rodando."
    );
  }

  if (!response.ok) {
    let detail = `A API retornou o status ${response.status}.`;
    try {
      const errorBody = await response.json();
      if (typeof errorBody.detail === "string") {
        detail = errorBody.detail;
      }
    } catch {
      // A resposta não contém JSON; usa a mensagem baseada no status.
    }
    throw new Error(detail);
  }

  const data: unknown = await response.json();
  if (!data || typeof data !== "object") {
    throw new Error("A API retornou uma resposta inválida.");
  }

  const parsed = data as RandomSimulationResponse;
  const prediction = parsed.prediction;
  if (
    !parsed.case ||
    !prediction ||
    !Array.isArray(prediction.models) ||
    typeof prediction.average !== "number" ||
    typeof prediction.isDengue !== "boolean"
  ) {
    throw new Error("A API retornou uma resposta inválida.");
  }

  return parsed;
}

export function avaliarDengue(
  selectedIds: string[],
  patientData: PatientData
): Promise<EvaluationResult> {
  return solicitarPredicao(montarPayload(selectedIds, patientData));
}
