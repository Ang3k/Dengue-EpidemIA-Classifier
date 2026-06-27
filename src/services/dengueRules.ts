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

// Acima deste valor (em %), o resultado final é considerado dengue
export const DENGUE_THRESHOLD = 40;

const API_URL = "http://localhost:8000";

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

// Monta o body que a API espera a partir dos dados do formulário
function montarPayload(
  selectedIds: string[],
  patientData: PatientData
): Record<string, unknown> {
  // Sintomas: 1 se marcado, 0 se não
  const sintomas: Record<string, number> = {};
  for (const item of triageItems) {
    sintomas[item.id] = selectedIds.includes(item.id) ? 1 : 0;
  }

  // Extrai mês e ano da data de notificação, se preenchida
  let notificationMonth: number | null = null;
  let notificationYear: number | null = null;
  if (patientData.notificationDate) {
    const d = new Date(patientData.notificationDate);
    notificationMonth = d.getMonth() + 1;
    notificationYear = d.getFullYear();
  } else {
    notificationMonth = patientData.notificationMonth
      ? Number(patientData.notificationMonth)
      : null;
    notificationYear = patientData.notificationYear
      ? Number(patientData.notificationYear)
      : null;
  }

  return {
    // Paciente
    age_years: patientData.ageYears
      ? Number(patientData.ageYears)
      : patientData.age
      ? Number(patientData.age)
      : null,
    sex: patientData.sex || null,
    pregnancy_status: patientData.pregnancyStatus
      ? Number(patientData.pregnancyStatus)
      : null,
    race: patientData.race ? Number(patientData.race) : null,
    education_level: patientData.educationLevel
      ? Number(patientData.educationLevel)
      : null,
    occupation_code: patientData.occupationCode || null,

    // Residência
    residence_state: patientData.residenceState
      ? Number(patientData.residenceState)
      : null,
    residence_municipality: patientData.residenceMunicipality
      ? Number(patientData.residenceMunicipality)
      : null,
    residence_health_region: patientData.residenceHealthRegion
      ? Number(patientData.residenceHealthRegion)
      : null,

    // Notificação
    notification_date: patientData.notificationDate || null,
    notification_year: notificationYear,
    notification_month: notificationMonth,
    notification_epi_week: patientData.notificationEpiWeek
      ? Number(patientData.notificationEpiWeek)
      : null,
    notif_municipality: patientData.notifMunicipality
      ? Number(patientData.notifMunicipality)
      : null,
    notif_health_region: patientData.notifHealthRegion
      ? Number(patientData.notifHealthRegion)
      : null,
    health_facility: patientData.healthFacility
      ? Number(patientData.healthFacility)
      : null,

    // Início dos sintomas
    symptom_onset_date: patientData.symptomOnsetDate || null,
    days_to_notification: patientData.daysToNotification
      ? Number(patientData.daysToNotification)
      : null,
    symptom_epi_year: patientData.symptomEpiYear
      ? Number(patientData.symptomEpiYear)
      : null,
    symptom_epi_week_number: patientData.symptomEpiWeekNumber
      ? Number(patientData.symptomEpiWeekNumber)
      : null,

    // Hospitalização
    hospitalized: patientData.hospitalized
      ? Number(patientData.hospitalized)
      : null,
    hospital_state: patientData.hospitalState
      ? Number(patientData.hospitalState)
      : null,

    // Sintomas
    ...sintomas,
  };
}

export async function avaliarDengue(
  selectedIds: string[],
  patientData: PatientData
): Promise<EvaluationResult> {
  const payload = montarPayload(selectedIds, patientData);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Erro na API: ${response.status}`);
  }

  const data = await response.json();

  return {
    models: data.models,
    average: data.average,
    isDengue: data.isDengue,
  };
}