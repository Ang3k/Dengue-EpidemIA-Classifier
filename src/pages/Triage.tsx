import { useState } from "react";
import CheckboxItem from "../components/CheckboxItem";
import PatientForm from "../components/PatientForm";
import Resultado from "../components/Resultado";
import { avaliarDengue, triageItems } from "../services/dengueRules";
import type { PatientData } from "../types/patient";

const grupos = [
  {
    id: "clinico",
    title: "Sinais clínicos",
  },
  {
    id: "comorbidade",
    title: "Doenças pré-existentes",
  },
  {
    id: "laboratorial",
    title: "Dados laboratoriais",
  },
  {
    id: "alarme",
    title: "Sinais de alarme",
  },
  {
    id: "grave",
    title: "Sinais de dengue grave",
  },
  {
    id: "hemorragico",
    title: "Manifestações hemorrágicas",
  },
];

function Triage() {
  const [patientData, setPatientData] = useState<PatientData>({
    idade: "",
    sexo: "",
    gestante: "",
    racaCor: "",
    escolaridade: "",
    ufResidencia: "",
    municipioResidencia: "",
    ufNotificacao: "",
    municipioNotificacao: "",
    dataPrimeirosSintomas: "",
    hospitalizado: "",
  });

  const [selectedItems, setSelectedItems] = useState<string[]>([]);

  function toggleItem(id: string) {
    setSelectedItems((current) => {
      if (current.includes(id)) {
        return current.filter((item) => item !== id);
      }

      return [...current, id];
    });
  }

  const resultado = avaliarDengue(selectedItems, patientData);

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

    <Resultado
  title={resultado.title}
  message={resultado.message}
  level={resultado.level}
  models={resultado.models}
/>
      </section>
    </main>
  );
}

export default Triage;