import { useState } from "react";
import CheckboxItem from "../components/CheckboxItem";
import PatientForm from "../components/PatientForm";
import Resultado from "../components/Resultado";
import { avaliarDengue, triageItems } from "../services/dengueRules";
import type { EvaluationResult } from "../services/dengueRules";
import type { PatientData } from "../types/patient";

const grupos = [
  {
    id: "symptoms",
    title: "Sintomas informados",
  },
];

function Triage() {
  const [patientData, setPatientData] = useState<PatientData>({
    age: "",
    ageYears: "",

    sex: "",
    sexLabel: "",

    pregnancyStatus: "",
    pregnancyStatusLabel: "",

    race: "",
    raceLabel: "",

    educationLevel: "",
    educationLevelLabel: "",

    occupationCode: "",
    occupationName: "",

    residenceState: "",
    residenceStateLabel: "",
    residenceMunicipality: "",
    residenceHealthRegion: "",

    diseaseCode: "",

    notificationDate: "",
    notificationYear: "",
    notificationMonth: "",
    notificationDay: "",
    notificationEpiWeek: "",

    notifMunicipality: "",
    notifHealthRegion: "",
    healthFacility: "",

    symptomOnsetDate: "",
    daysToNotification: "",
    symptomEpiYear: "",
    symptomEpiWeekNumber: "",

    hospitalized: "",
    hospitalState: "",
    hospitalStateLabel: "",
  });

  const [selectedItems, setSelectedItems] = useState<string[]>([]);
  const [resultado, setResultado] = useState<EvaluationResult | null>(null);
  const [carregando, setCarregando] = useState(false);
  const [erro, setErro] = useState<string | null>(null);

  function toggleItem(id: string) {
    setSelectedItems((current) => {
      if (current.includes(id)) {
        return current.filter((item) => item !== id);
      }
      return [...current, id];
    });
  }

  async function handleEnviarTriagem() {
    setCarregando(true);
    setErro(null);
    setResultado(null);

    try {
      const resultadoFinal = await avaliarDengue(selectedItems, patientData);
      setResultado(resultadoFinal);
    } catch (e) {
      setErro(
        "Não foi possível conectar à API. Verifique se o servidor está rodando."
      );
    } finally {
      setCarregando(false);
    }
  }

  return (
    <main className="container">
      <section className="card">
        <h1>Triagem de Dengue</h1>

        <p>
          Preencha os dados principais do paciente e marque os sinais, sintomas
          e condições abaixo. O sistema fará uma triagem baseada nos principais
          campos usados na ficha de dengue do Sinan.
        </p>

        <PatientForm
          patientData={patientData}
          setPatientData={setPatientData}
        />

        {grupos.map((grupo) => {
          const itensDoGrupo = triageItems.filter(
            (item) => item.group === grupo.id
          );

          return (
            <section className="grupo-sintomas" key={grupo.id}>
              <h2>{grupo.title}</h2>

              <div className="checkbox-list">
                {itensDoGrupo.map((item) => (
                  <CheckboxItem
                    key={item.id}
                    label={item.label}
                    checked={selectedItems.includes(item.id)}
                    onChange={() => toggleItem(item.id)}
                  />
                ))}
              </div>
            </section>
          );
        })}

        <div className="actions">
          <button
            type="button"
            className="btn-primary"
            onClick={handleEnviarTriagem}
            disabled={carregando}
          >
            {carregando ? "Calculando..." : "Enviar triagem"}
          </button>
        </div>

        {erro && (
          <p style={{ color: "red", marginTop: "1rem" }}>{erro}</p>
        )}

        {resultado && (
          <Resultado
            models={resultado.models}
            average={resultado.average}
            isDengue={resultado.isDengue}
          />
        )}
      </section>
    </main>
  );
}

export default Triage;