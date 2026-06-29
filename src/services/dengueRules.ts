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
  weight: number;
};

export type EvaluationResult = {
  models: ModelPrediction[];
  average: number;
  threshold: number;
  weighting: "recall";
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

const API_URL = (
  import.meta.env.VITE_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

const MODEL_LABELS: Record<string, string> = {
  mlp: "MLP",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
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

function mensagemErroApi(errorBody: unknown, status: number): string {
  if (!errorBody || typeof errorBody !== "object") {
    return `A API retornou o status ${status}.`;
  }

  const detail = (errorBody as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return `A API retornou o status ${status}.`;

  const messages = detail.flatMap(issue => {
    if (!issue || typeof issue !== "object") return [];
    const { loc, msg } = issue as { loc?: unknown[]; msg?: unknown };
    if (typeof msg !== "string") return [];
    const field = Array.isArray(loc) ? String(loc.at(-1) ?? "") : "";
    return [field ? `${field}: ${msg}` : msg];
  });

  return messages.length > 0
    ? messages.join(" | ")
    : `A API retornou o status ${status}.`;
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
      detail = mensagemErroApi(errorBody, response.status);
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
    typeof (data as EvaluationResult).threshold !== "number" ||
    (data as EvaluationResult).weighting !== "recall" ||
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
      detail = mensagemErroApi(errorBody, response.status);
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
    typeof prediction.threshold !== "number" ||
    prediction.weighting !== "recall" ||
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
  if (
    patientData.notificationDate &&
    patientData.symptomOnsetDate &&
    patientData.notificationDate < patientData.symptomOnsetDate
  ) {
    return Promise.reject(
      new Error(
        "A data da notificação não pode ser anterior ao início dos sintomas."
      )
    );
  }

  return solicitarPredicao(montarPayload(selectedIds, patientData));
}
