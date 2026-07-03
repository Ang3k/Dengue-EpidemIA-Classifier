import { useEffect, useRef, useState } from "react";
import type { PatientData } from "../types/patient";

const API_URL = "http://localhost:8000";

// ---------------------------------------------------------------------------
// Tipos
// ---------------------------------------------------------------------------

type SelectOption = { code: number | string; name: string };
type UfOption = { code: number; sigla: string; name: string };
type MunicipioItem = { code: number; name: string; stateCode: number; state: string };
type RegiaoItem = { code: number; name: string; state: string };
type AutocompleteItem = { code: number | string; name: string; state?: string; stateCode?: number };

type TriageOptions = {
  sexos: SelectOption[];
  racas: SelectOption[];
  escolaridades: SelectOption[];
  situacoesGestacao: SelectOption[];
  ufs: UfOption[];
};

// ---------------------------------------------------------------------------
// Semana epidemiológica (padrão SINAN: semana começa no domingo)
// ---------------------------------------------------------------------------

function calcularSemanaEpi(data: Date): { semana: number; ano: number } {
  // Primeiro dia do ano
  const inicio = new Date(data.getFullYear(), 0, 1);
  const diaSemana = inicio.getDay(); // 0 = domingo
  // Recua até o domingo anterior (ou fica no mesmo se já for domingo)
  const primeiroDomingo = new Date(inicio);
  primeiroDomingo.setDate(inicio.getDate() - diaSemana);
  const diff = Math.floor(
    (data.getTime() - primeiroDomingo.getTime()) / (7 * 24 * 60 * 60 * 1000)
  );
  const semana = diff + 1;
  // Se cair antes da semana 1, pertence ao último período do ano anterior
  if (semana < 1) {
    return calcularSemanaEpi(new Date(data.getFullYear() - 1, 11, 31));
  }
  return { semana, ano: data.getFullYear() };
}

// ---------------------------------------------------------------------------
// Hook de autocomplete com debounce
// ---------------------------------------------------------------------------

function useAutocomplete(
  fetchFn: (q: string) => Promise<AutocompleteItem[]>,
  delay = 300
) {
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<AutocompleteItem[]>([]);
  const [aberto, setAberto] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (query.length < 2) {
      setItems([]);
      setAberto(false);
      return;
    }
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      const resultado = await fetchFn(query);
      setItems(resultado);
      setAberto(resultado.length > 0);
    }, delay);
    return () => { if (timer.current) clearTimeout(timer.current); };
  }, [query]);

  return { query, setQuery, items, aberto, setAberto };
}

// ---------------------------------------------------------------------------
// Componente Autocomplete
// ---------------------------------------------------------------------------

type AutocompleteProps = {
  label: string;
  id: string;
  placeholder: string;
  fetchFn: (q: string) => Promise<AutocompleteItem[]>;
  onSelect: (item: AutocompleteItem) => void;
  renderLabel?: (item: AutocompleteItem) => string;
  displayValue: string;
};

function Autocomplete({
  label, id, placeholder, fetchFn, onSelect, renderLabel, displayValue,
}: AutocompleteProps) {
  const { query, setQuery, items, aberto, setAberto } = useAutocomplete(fetchFn);
  const containerRef = useRef<HTMLDivElement>(null);
  const [focusIndex, setFocusIndex] = useState(-1);

  // Fecha ao clicar fora
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setAberto(false);
      }
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Sincroniza texto exibido com valor externo (ex: ao limpar o form)
  useEffect(() => {
    if (!displayValue) setQuery("");
  }, [displayValue]);

  function handleSelect(item: AutocompleteItem) {
    onSelect(item);
    setQuery(renderLabel ? renderLabel(item) : item.name);
    setAberto(false);
    setFocusIndex(-1);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (!aberto) return;
    if (e.key === "ArrowDown") { e.preventDefault(); setFocusIndex(i => Math.min(i + 1, items.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setFocusIndex(i => Math.max(i - 1, 0)); }
    else if (e.key === "Enter" && focusIndex >= 0) { e.preventDefault(); handleSelect(items[focusIndex]); }
    else if (e.key === "Escape") setAberto(false);
  }

  return (
    <div className="autocomplete-wrapper" ref={containerRef}>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="text"
        autoComplete="off"
        placeholder={placeholder}
        value={query}
        onChange={e => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      {aberto && (
        <ul className="autocomplete-list" role="listbox">
          {items.map((item, idx) => (
            <li
              key={item.code}
              role="option"
              aria-selected={idx === focusIndex}
              className={`autocomplete-item${idx === focusIndex ? " focused" : ""}`}
              onMouseDown={() => handleSelect(item)}
            >
              {renderLabel ? renderLabel(item) : item.name}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

async function fetchOcupacoes(query: string): Promise<AutocompleteItem[]> {
  const res = await fetch(
    `${API_URL}/api/v1/references/occupations?query=${encodeURIComponent(query)}&limit=10`
  );
  if (!res.ok) return [];
  return (await res.json()).items as AutocompleteItem[];
}

function makeFetchMunicipios(stateCode?: number) {
  return async (query: string): Promise<AutocompleteItem[]> => {
    const params = new URLSearchParams({ query, limit: "20" });
    if (stateCode) params.set("state", String(stateCode));
    const res = await fetch(`${API_URL}/api/v1/references/municipalities?${params}`);
    if (!res.ok) return [];
    return (await res.json()).items as MunicipioItem[];
  };
}

// ---------------------------------------------------------------------------
// PatientForm
// ---------------------------------------------------------------------------

type PatientFormProps = {
  patientData: PatientData;
  setPatientData: React.Dispatch<React.SetStateAction<PatientData>>;
};

function PatientForm({ patientData, setPatientData }: PatientFormProps) {
  const [options, setOptions] = useState<TriageOptions | null>(null);
  const [, setRegioesResidencia] = useState<RegiaoItem[]>([]);

  // Carrega opções da API uma vez
  useEffect(() => {
    fetch(`${API_URL}/api/v1/triage/options`)
      .then(r => r.json())
      .then((data) =>
        setOptions({
          sexos: data.sexos,
          racas: data.racas,
          escolaridades: data.escolaridades,
          situacoesGestacao: data.situacoesGestacao,
          ufs: data.ufs,
        })
      )
      .catch(() => {});
  }, []);

  // Recalcula semana epi e dias ao mudar datas
  useEffect(() => {
    const onset = patientData.symptomOnsetDate
      ? new Date(patientData.symptomOnsetDate + "T00:00:00")
      : null;
    const notif = patientData.notificationDate
      ? new Date(patientData.notificationDate + "T00:00:00")
      : null;

    if (onset) {
      const { semana, ano } = calcularSemanaEpi(onset);
      setPatientData(prev => ({
        ...prev,
        symptomEpiWeekNumber: String(semana),
        symptomEpiYear: String(ano),
      }));
    }

    if (onset && notif && notif >= onset) {
      const dias = Math.round(
        (notif.getTime() - onset.getTime()) / (1000 * 60 * 60 * 24)
      );
      setPatientData(prev => ({ ...prev, daysToNotification: String(dias) }));
    }
  }, [patientData.symptomOnsetDate, patientData.notificationDate]);

  function set(field: keyof PatientData, value: string) {
    setPatientData(prev => ({ ...prev, [field]: value }));
  }

  async function aoSelecionarMunicipio(item: AutocompleteItem) {
    const mun = item as MunicipioItem;
    const uf = options?.ufs.find(u => u.code === mun.stateCode);
    setPatientData(prev => ({
      ...prev,
      residenceMunicipality: String(mun.code),
      residenceState: mun.stateCode ? String(mun.stateCode) : prev.residenceState,
      residenceStateLabel: uf?.sigla ?? prev.residenceStateLabel,
    }));

    try {
      const res = await fetch(
        `${API_URL}/api/v1/references/health-regions?municipality=${mun.code}`
      );
      const data = await res.json();
      const regioes: RegiaoItem[] = data.items ?? [];
      setRegioesResidencia(regioes);
      if (regioes.length === 1) {
        set("residenceHealthRegion", String(regioes[0].code));
      } else if (regioes.length === 0) {
        set("residenceHealthRegion", "");
      }
    } catch {
      setRegioesResidencia([]);
    }
  }

  const selectedStateCode = patientData.residenceState
    ? Number(patientData.residenceState)
    : undefined;

  return (
    <section className="patient-form">
      <h2>Dados usados pelo modelo</h2>

      <div className="form-grid">

        {/* Idade */}
        <div className="form-group">
          <label htmlFor="ageYears">Idade (anos)</label>
          <input
            id="ageYears"
            type="number"
            value={patientData.ageYears}
            onChange={e => set("ageYears", e.target.value)}
            min="0" max="130" step="1"
            placeholder="Ex.: 25"
          />
        </div>

        {/* Sexo */}
        <div className="form-group">
          <label htmlFor="sex">Sexo</label>
          <select id="sex" value={patientData.sex} onChange={e => set("sex", e.target.value)}>
            <option value="">Selecione</option>
            {(options?.sexos ?? []).map(s => (
              <option key={s.code} value={s.code}>{s.name}</option>
            ))}
          </select>
        </div>

        {/* Gestação */}
        <div className="form-group">
          <label htmlFor="pregnancyStatus">Situação de gestação</label>
          <select id="pregnancyStatus" value={patientData.pregnancyStatus} onChange={e => set("pregnancyStatus", e.target.value)}>
            <option value="">Selecione</option>
            {(options?.situacoesGestacao ?? []).map(g => (
              <option key={g.code} value={g.code}>{g.name}</option>
            ))}
          </select>
        </div>

        {/* Raça */}
        <div className="form-group">
          <label htmlFor="race">Raça/cor</label>
          <select id="race" value={patientData.race} onChange={e => set("race", e.target.value)}>
            <option value="">Selecione</option>
            {(options?.racas ?? []).map(r => (
              <option key={r.code} value={r.code}>{r.name}</option>
            ))}
          </select>
        </div>

        {/* Escolaridade */}
        <div className="form-group">
          <label htmlFor="educationLevel">Escolaridade</label>
          <select id="educationLevel" value={patientData.educationLevel} onChange={e => set("educationLevel", e.target.value)}>
            <option value="">Selecione</option>
            {(options?.escolaridades ?? []).map(e => (
              <option key={e.code} value={e.code}>{e.name}</option>
            ))}
          </select>
        </div>

        {/* Ocupação: autocomplete */}
        <div className="form-group form-group-wide">
          <Autocomplete
            id="occupationName"
            label="Ocupação"
            placeholder="Digite para buscar (ex: médico, professor...)"
            fetchFn={fetchOcupacoes}
            displayValue={patientData.occupationName ?? ""}
            onSelect={item => {
              set("occupationCode", String(item.code));
              set("occupationName", item.name);
            }}
          />
          {patientData.occupationCode && (
            <span className="form-hint">CBO: {patientData.occupationCode}</span>
          )}
        </div>

        {/* UF de residência */}
        <div className="form-group">
          <label htmlFor="residenceState">UF de residência</label>
          <select
            id="residenceState"
            value={patientData.residenceState}
            onChange={e => {
              const uf = options?.ufs.find(u => String(u.code) === e.target.value);
              setPatientData(prev => ({
                ...prev,
                residenceState: e.target.value,
                residenceStateLabel: uf?.sigla ?? "",
                residenceMunicipality: "",
                residenceHealthRegion: "",
              }));
              setRegioesResidencia([]);
            }}
          >
            <option value="">Selecione</option>
            {(options?.ufs ?? []).map(uf => (
              <option key={uf.code} value={uf.code}>
                {uf.sigla} ({uf.name})
              </option>
            ))}
          </select>
        </div>

        {/* Município de residência: autocomplete */}
        <div className="form-group form-group-wide">
          <Autocomplete
            id="residenceMunicipality"
            label="Município de residência"
            placeholder="Digite para buscar (ex: Rio de Janeiro...)"
            fetchFn={makeFetchMunicipios(selectedStateCode)}
            displayValue={patientData.residenceMunicipality}
            onSelect={aoSelecionarMunicipio}
            renderLabel={item =>
              item.state ? `${item.name} (${item.state})` : item.name
            }
          />
          {patientData.residenceMunicipality && (
            <span className="form-hint">IBGE: {patientData.residenceMunicipality}</span>
          )}
        </div>

        {/* Região de saúde: preenchida automaticamente ou select se houver mais de uma */}
       

        {/* Data dos primeiros sintomas */}
        <div className="form-group">
          <label htmlFor="symptomOnsetDate">Data dos primeiros sintomas</label>
          <input
            id="symptomOnsetDate"
            type="date"
            value={patientData.symptomOnsetDate}
            onChange={e => set("symptomOnsetDate", e.target.value)}
          />
        </div>

        {/* Data da notificação */}
        <div className="form-group">
          <label htmlFor="notificationDate">Data da notificação</label>
          <input
            id="notificationDate"
            type="date"
            value={patientData.notificationDate}
            onChange={e => set("notificationDate", e.target.value)}
          />
        </div>

      </div>
    </section>
  );
}

export default PatientForm;
