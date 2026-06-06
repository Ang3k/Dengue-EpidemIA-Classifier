import type { PatientData } from "../types/patient";

export type TriageItem = {
  id: string;
  label: string;
  points: number;
  group: "clinico" | "comorbidade" | "alarme" | "grave" | "hemorragico" | "laboratorial";
  danger?: boolean;
};

export const triageItems: TriageItem[] = [
  // SINAIS CLÍNICOS PRINCIPAIS - SINAN
  {
    id: "febre",
    label: "Febre",
    points: 3,
    group: "clinico",
  },
  {
    id: "mialgia",
    label: "Mialgia / dor muscular",
    points: 2,
    group: "clinico",
  },
  {
    id: "cefaleia",
    label: "Cefaleia / dor de cabeça",
    points: 2,
    group: "clinico",
  },
  {
    id: "exantema",
    label: "Exantema / manchas vermelhas na pele",
    points: 2,
    group: "clinico",
  },
  {
    id: "vomito",
    label: "Vômito",
    points: 2,
    group: "clinico",
  },
  {
    id: "nausea",
    label: "Náusea / enjoo",
    points: 1,
    group: "clinico",
  },
  {
    id: "dor_costas",
    label: "Dor nas costas",
    points: 1,
    group: "clinico",
  },
  {
    id: "conjuntivite",
    label: "Conjuntivite",
    points: 1,
    group: "clinico",
  },
  {
    id: "artrite",
    label: "Artrite / inflamação nas articulações",
    points: 1,
    group: "clinico",
  },
  {
    id: "artralgia",
    label: "Artralgia intensa / dor forte nas articulações",
    points: 2,
    group: "clinico",
  },
  {
    id: "petequias",
    label: "Petéquias / pequenos pontos vermelhos na pele",
    points: 3,
    group: "clinico",
  },
  {
    id: "leucopenia",
    label: "Leucopenia em exame de sangue",
    points: 3,
    group: "laboratorial",
  },
  {
    id: "prova_laco",
    label: "Prova do laço positiva",
    points: 3,
    group: "laboratorial",
  },
  {
    id: "dor_retroorbital",
    label: "Dor atrás dos olhos / dor retro-orbital",
    points: 2,
    group: "clinico",
  },

  // DOENÇAS PRÉ-EXISTENTES
  {
    id: "diabetes",
    label: "Diabetes",
    points: 1,
    group: "comorbidade",
  },
  {
    id: "hematologica",
    label: "Doença hematológica",
    points: 1,
    group: "comorbidade",
  },
  {
    id: "hepatopatia",
    label: "Hepatopatia / doença no fígado",
    points: 1,
    group: "comorbidade",
  },
  {
    id: "renal",
    label: "Doença renal crônica",
    points: 1,
    group: "comorbidade",
  },
  {
    id: "hipertensao",
    label: "Hipertensão arterial",
    points: 1,
    group: "comorbidade",
  },
  {
    id: "acido_peptica",
    label: "Doença ácido-péptica / gastrite ou úlcera",
    points: 1,
    group: "comorbidade",
  },
  {
    id: "autoimune",
    label: "Doença autoimune",
    points: 1,
    group: "comorbidade",
  },

  // SINAIS DE ALARME
  {
    id: "alarme_hipotensao",
    label: "Hipotensão / pressão baixa",
    points: 8,
    group: "alarme",
    danger: true,
  },
  {
    id: "alarme_plaquetas",
    label: "Queda abrupta de plaquetas",
    points: 8,
    group: "alarme",
    danger: true,
  },
  {
    id: "alarme_vomitos",
    label: "Vômitos persistentes",
    points: 8,
    group: "alarme",
    danger: true,
  },
  {
    id: "alarme_sangramento",
    label: "Sangramento de mucosa ou outras hemorragias",
    points: 8,
    group: "alarme",
    danger: true,
  },
  {
    id: "alarme_hematocrito",
    label: "Aumento do hematócrito",
    points: 8,
    group: "alarme",
    danger: true,
  },
  {
    id: "alarme_abdominal",
    label: "Dor abdominal intensa",
    points: 8,
    group: "alarme",
    danger: true,
  },
  {
    id: "alarme_letargia",
    label: "Letargia ou irritabilidade",
    points: 8,
    group: "alarme",
    danger: true,
  },
  {
    id: "alarme_hepatomegalia",
    label: "Hepatomegalia / fígado aumentado",
    points: 8,
    group: "alarme",
    danger: true,
  },
  {
    id: "alarme_liquidos",
    label: "Acúmulo de líquidos",
    points: 8,
    group: "alarme",
    danger: true,
  },

  // SINAIS DE DENGUE GRAVE
  {
    id: "grave_pulso",
    label: "Pulso débil ou indetectável",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_pa_convergente",
    label: "Pressão arterial convergente",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_enchimento",
    label: "Alteração no tempo de enchimento capilar",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_insuf_respiratoria",
    label: "Acúmulo de líquidos com insuficiência respiratória",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_taquicardia",
    label: "Taquicardia",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_extremidades",
    label: "Extremidades frias",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_hipotensao_tardia",
    label: "Hipotensão arterial em fase tardia",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_hematemese",
    label: "Hematêmese / vômito com sangue",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_melena",
    label: "Melena / fezes escuras com sangue",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_metrorragia",
    label: "Metrorragia volumosa",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_sangramento_snc",
    label: "Sangramento do sistema nervoso central",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_ast_alt",
    label: "AST/ALT maior que 1000",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_miocardite",
    label: "Miocardite",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_consciencia",
    label: "Alteração da consciência",
    points: 12,
    group: "grave",
    danger: true,
  },
  {
    id: "grave_outros_orgaos",
    label: "Comprometimento de outros órgãos",
    points: 12,
    group: "grave",
    danger: true,
  },

  // MANIFESTAÇÕES HEMORRÁGICAS
  {
    id: "mani_hemorragica",
    label: "Manifestações hemorrágicas",
    points: 6,
    group: "hemorragico",
    danger: true,
  },
  {
    id: "epistaxe",
    label: "Epistaxe / sangramento nasal",
    points: 6,
    group: "hemorragico",
    danger: true,
  },
  {
    id: "gengivorragia",
    label: "Gengivorragia / sangramento na gengiva",
    points: 6,
    group: "hemorragico",
    danger: true,
  },
  {
    id: "metrorragia",
    label: "Metrorragia",
    points: 6,
    group: "hemorragico",
    danger: true,
  },
  {
    id: "hematuria",
    label: "Hematúria / sangue na urina",
    points: 6,
    group: "hemorragico",
    danger: true,
  },
  {
    id: "sangramento_gastro",
    label: "Sangramento gastrointestinal",
    points: 8,
    group: "hemorragico",
    danger: true,
  },
  {
    id: "extravasamento",
    label: "Extravasamento plasmático",
    points: 8,
    group: "hemorragico",
    danger: true,
  },
];

export function avaliarDengue(selectedIds: string[], patientData: PatientData) {
  const selectedItems = triageItems.filter((item) =>
    selectedIds.includes(item.id)
  );

  const totalPoints = selectedItems.reduce((sum, item) => sum + item.points, 0);

  const hasFever = selectedIds.includes("febre");

  const clinicalCount = selectedItems.filter(
    (item) => item.group === "clinico"
  ).length;

  const comorbidityCount = selectedItems.filter(
    (item) => item.group === "comorbidade"
  ).length;

  const hasAlarmSign = selectedItems.some((item) => item.group === "alarme");

  const hasSevereSign = selectedItems.some((item) => item.group === "grave");

  const hasBleeding = selectedItems.some(
    (item) => item.group === "hemorragico"
  );
  const idade = Number(patientData.idade);

const isCrianca = idade > 0 && idade < 12;
const isIdoso = idade >= 60;
const isGestante =
  patientData.gestante === "1" ||
  patientData.gestante === "2" ||
  patientData.gestante === "3" ||
  patientData.gestante === "4";

const hasComorbidity = selectedItems.some(
  (item) => item.group === "comorbidade"
);

const grupoRisco = isCrianca || isIdoso || isGestante || hasComorbidity;

  if (hasSevereSign) {
   return {
  level: "GRAVE",
  title: "Possíveis sinais de dengue grave",
  message:
    "Você marcou sinais compatíveis com dengue grave. Procure atendimento médico imediatamente ou uma emergência.",
  models: {
    clinical: {
      name: "Modelo Clínico",
      result: "Risco muito alto",
      confidence: "94%",
      description:
        "Identificou sinais compatíveis com dengue grave, como alterações circulatórias, sangramentos importantes ou comprometimento de órgãos.",
    },
    epidemiological: {
      name: "Modelo Epidemiológico",
      result: "Prioridade máxima",
      confidence: "89%",
      description:
        "Considerou fatores como idade, comorbidades, gestação, município informado e necessidade de atendimento imediato.",
    },
  },
};
  }

  if (hasAlarmSign || hasBleeding) {
    return {
  level: "ALERTA",
  title: "Sinais de alarme para dengue grave",
  message:
    "Você marcou sinais de alarme ou sangramento. Procure atendimento médico com urgência.",
  models: {
    clinical: {
      name: "Modelo Clínico",
      result: "Alerta clínico",
      confidence: "88%",
      description:
        "Encontrou sinais de alarme, como dor abdominal intensa, vômitos persistentes, sangramentos ou queda de plaquetas.",
    },
    epidemiological: {
      name: "Modelo Epidemiológico",
      result: "Risco aumentado",
      confidence: "81%",
      description:
        "Avaliou os dados do paciente e indicou maior necessidade de acompanhamento pela unidade de saúde.",
    },
  },
};
  }

if (hasFever && clinicalCount >= 3) {
  return {
    level: "ALTA",
    title: "Alta suspeita de dengue",
    message:
      "Você marcou febre e vários sinais clínicos compatíveis com dengue. Procure uma unidade de saúde para avaliação.",
    models: {
      clinical: {
        name: "Modelo Clínico",
        result: "Alta compatibilidade",
        confidence: "83%",
        description:
          "O modelo clínico identificou febre associada a vários sinais compatíveis com dengue, como dor no corpo, cefaleia, dor atrás dos olhos, náusea, manchas na pele ou dores articulares.",
      },
      epidemiological: {
        name: "Modelo Epidemiológico",
        result: "Risco moderado a alto",
        confidence: "72%",
        description:
          "O modelo epidemiológico simulou a análise de idade, município, data dos primeiros sintomas, comorbidades e demais dados do paciente para complementar a triagem.",
      },
    },
  };
}

  if (hasFever && clinicalCount >= 2) {
   return {
  level: "MODERADA",
  title: "Suspeita moderada de dengue",
  message:
    "Você marcou febre e alguns sintomas compatíveis com dengue. Observe a evolução e procure atendimento, principalmente se houver piora.",
  models: {
    clinical: {
      name: "Modelo Clínico",
      result: "Compatibilidade parcial",
      confidence: "68%",
      description:
        "Há sintomas compatíveis com dengue, mas a combinação marcada ainda não é suficiente para indicar alta suspeita.",
    },
    epidemiological: {
      name: "Modelo Epidemiológico",
      result: "Risco variável",
      confidence: "64%",
      description:
        "Os dados pessoais e territoriais sugerem acompanhamento, principalmente se houver casos recentes no município.",
    },
  },
};
  }

  if (totalPoints >= 7 || comorbidityCount >= 2) {
  return {
    level: "MODERADA",
    title: "Atenção aos sintomas",
    message:
      "Há sintomas ou condições pré-existentes que merecem atenção. O resultado não confirma dengue, mas recomenda acompanhamento.",
    models: {
      clinical: {
        name: "Modelo Clínico",
        result: "Compatibilidade parcial",
        confidence: "67%",
        description:
          "O modelo clínico identificou sintomas compatíveis com dengue, mas sem sinais suficientes para classificar como alta suspeita ou dengue grave.",
      },
      epidemiological: {
        name: "Modelo Epidemiológico",
        result: "Atenção ao perfil do paciente",
        confidence: "74%",
        description:
          "O modelo epidemiológico considerou condições pré-existentes, idade, município informado e dados gerais do paciente como fatores que podem exigir acompanhamento.",
      },
    },
  };
}
 if (grupoRisco && totalPoints >= 5) {
  return {
    level: "MODERADA",
    title: "Atenção: paciente com fator de risco",
    message:
      "Além dos sintomas marcados, há fator de risco como idade, gestação ou doença pré-existente. Recomenda-se procurar uma unidade de saúde para avaliação.",
    models: {
      clinical: {
        name: "Modelo Clínico",
        result: "Compatibilidade parcial",
        confidence: "70%",
        description:
          "O modelo clínico identificou sintomas compatíveis com dengue, mas sem sinais suficientes para classificar como dengue grave.",
      },
      epidemiological: {
        name: "Modelo Epidemiológico",
        result: "Risco aumentado",
        confidence: "76%",
        description:
          "O modelo epidemiológico considerou fatores como idade, gestação, doenças pré-existentes, município informado e data dos primeiros sintomas.",
      },
    },
  };
}
  return {
  level: "BAIXA",
  title: "Baixa suspeita de dengue",
  message:
    "Pelos itens marcados, a suspeita parece baixa. Mesmo assim, se houver febre persistente, piora ou novos sintomas, procure atendimento.",
  models: {
    clinical: {
      name: "Modelo Clínico",
      result: "Baixa compatibilidade",
      confidence: "61%",
      description:
        "Poucos sintomas típicos foram marcados, então a compatibilidade clínica com dengue ficou baixa.",
    },
    epidemiological: {
      name: "Modelo Epidemiológico",
      result: "Baixo risco informado",
      confidence: "58%",
      description:
        "Com os dados preenchidos, não foram simulados fatores suficientes para elevar o risco epidemiológico.",
    },
  },
};
}