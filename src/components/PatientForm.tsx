import type { PatientData } from "../types/patient";

type PatientFormProps = {
  patientData: PatientData;
  setPatientData: React.Dispatch<React.SetStateAction<PatientData>>;
};

const UFS = [
  ["11", "RO"], ["12", "AC"], ["13", "AM"], ["14", "RR"],
  ["15", "PA"], ["16", "AP"], ["17", "TO"], ["21", "MA"],
  ["22", "PI"], ["23", "CE"], ["24", "RN"], ["25", "PB"],
  ["26", "PE"], ["27", "AL"], ["28", "SE"], ["29", "BA"],
  ["31", "MG"], ["32", "ES"], ["33", "RJ"], ["35", "SP"],
  ["41", "PR"], ["42", "SC"], ["43", "RS"], ["50", "MS"],
  ["51", "MT"], ["52", "GO"], ["53", "DF"],
] as const;

function PatientForm({ patientData, setPatientData }: PatientFormProps) {
  function handleChange(
    event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) {
    const { name, value } = event.target;
    setPatientData((current) => ({ ...current, [name]: value }));
  }

  return (
    <section className="patient-form">
      <h2>Dados usados pelo modelo</h2>

      <div className="form-grid">
        <div className="form-group">
          <label htmlFor="ageYears">Idade (anos)</label>
          <input
            id="ageYears"
            type="number"
            name="ageYears"
            value={patientData.ageYears}
            onChange={handleChange}
            min="0"
            max="130"
            step="1"
            placeholder="Ex.: 25"
          />
        </div>

        <div className="form-group">
          <label htmlFor="sex">Sexo</label>
          <select
            id="sex"
            name="sex"
            value={patientData.sex}
            onChange={handleChange}
          >
            <option value="">Selecione</option>
            <option value="M">Masculino</option>
            <option value="F">Feminino</option>
            <option value="I">Ignorado</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="pregnancyStatus">Status de gestação</label>
          <select
            id="pregnancyStatus"
            name="pregnancyStatus"
            value={patientData.pregnancyStatus}
            onChange={handleChange}
          >
            <option value="">Selecione</option>
            <option value="1">1º trimestre</option>
            <option value="2">2º trimestre</option>
            <option value="3">3º trimestre</option>
            <option value="4">Idade gestacional ignorada</option>
            <option value="5">Não</option>
            <option value="6">Não se aplica</option>
            <option value="9">Ignorado</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="race">Raça/cor</label>
          <select
            id="race"
            name="race"
            value={patientData.race}
            onChange={handleChange}
          >
            <option value="">Selecione</option>
            <option value="1">Branca</option>
            <option value="2">Preta</option>
            <option value="3">Amarela</option>
            <option value="4">Parda</option>
            <option value="5">Indígena</option>
            <option value="9">Ignorado</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="educationLevel">Escolaridade</label>
          <select
            id="educationLevel"
            name="educationLevel"
            value={patientData.educationLevel}
            onChange={handleChange}
          >
            <option value="">Selecione</option>
            <option value="0">Analfabeto</option>
            <option value="1">1ª a 4ª série incompleta</option>
            <option value="2">4ª série completa</option>
            <option value="3">5ª a 8ª série incompleta</option>
            <option value="4">Ensino fundamental completo</option>
            <option value="5">Ensino médio incompleto</option>
            <option value="6">Ensino médio completo</option>
            <option value="7">Superior incompleto</option>
            <option value="8">Superior completo</option>
            <option value="10">Não se aplica</option>
            <option value="9">Ignorado</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="occupationCode">Código CBO da ocupação</label>
          <input
            id="occupationCode"
            type="text"
            inputMode="numeric"
            name="occupationCode"
            value={patientData.occupationCode}
            onChange={handleChange}
            pattern="[0-9]{5,6}"
            maxLength={6}
            placeholder="Ex.: 225125"
          />
        </div>

        <div className="form-group">
          <label htmlFor="residenceState">UF de residência</label>
          <select
            id="residenceState"
            name="residenceState"
            value={patientData.residenceState}
            onChange={handleChange}
          >
            <option value="">Selecione</option>
            {UFS.map(([code, label]) => (
              <option key={code} value={code}>
                {label}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="residenceMunicipality">
            Código IBGE do município
          </label>
          <input
            id="residenceMunicipality"
            type="text"
            inputMode="numeric"
            name="residenceMunicipality"
            value={patientData.residenceMunicipality}
            onChange={handleChange}
            pattern="[0-9]{7}"
            maxLength={7}
            placeholder="Ex.: 3304557"
          />
        </div>

        <div className="form-group">
          <label htmlFor="residenceHealthRegion">
            Código da região de saúde
          </label>
          <input
            id="residenceHealthRegion"
            type="text"
            inputMode="numeric"
            name="residenceHealthRegion"
            value={patientData.residenceHealthRegion}
            onChange={handleChange}
            pattern="[0-9]+"
            placeholder="Código numérico"
          />
        </div>

        <div className="form-group">
          <label htmlFor="notificationDate">Data da notificação</label>
          <input
            id="notificationDate"
            type="date"
            name="notificationDate"
            value={patientData.notificationDate}
            onChange={handleChange}
          />
        </div>

        <div className="form-group">
          <label htmlFor="symptomOnsetDate">
            Data dos primeiros sintomas
          </label>
          <input
            id="symptomOnsetDate"
            type="date"
            name="symptomOnsetDate"
            value={patientData.symptomOnsetDate}
            onChange={handleChange}
          />
        </div>

        <div className="form-group">
          <label htmlFor="daysToNotification">
            Dias até a notificação
          </label>
          <input
            id="daysToNotification"
            type="number"
            name="daysToNotification"
            value={patientData.daysToNotification}
            onChange={handleChange}
            min="0"
            max="90"
            placeholder="Calculado pelas datas se ficar vazio"
          />
        </div>

        <div className="form-group">
          <label htmlFor="symptomEpiWeekNumber">
            Semana epidemiológica dos sintomas
          </label>
          <input
            id="symptomEpiWeekNumber"
            type="number"
            name="symptomEpiWeekNumber"
            value={patientData.symptomEpiWeekNumber}
            onChange={handleChange}
            min="1"
            max="53"
            placeholder="Ex.: 12"
          />
        </div>
      </div>
    </section>
  );
}

export default PatientForm;
