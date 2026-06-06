import { Link } from "react-router-dom";

function Graphics() {
  return (
    <main className="container">
      <section className="card">
        <h1>Análise dos Dados</h1>

        <p>
          Nesta página serão exibidas análises gráficas e textuais com insights
          obtidos ao longo do projeto. Os gráficos abaixo são placeholders —
          substitua pelas visualizações geradas a partir dos dados reais.
        </p>

        <div style={{ display: "grid", gap: 16, marginTop: 16 }}>
          <div style={{ background: "#f3f3f3", height: 220, display: "flex", alignItems: "center", justifyContent: "center" }}>
            Placeholder: Gráfico 1 (ex.: Distribuição de sintomas por idade)
          </div>

          <div style={{ background: "#f3f3f3", height: 220, display: "flex", alignItems: "center", justifyContent: "center" }}>
            Placeholder: Gráfico 2 (ex.: Tendência temporal de casos suspeitos)
          </div>

          <div style={{ background: "#f3f3f3", height: 220, display: "flex", alignItems: "center", justifyContent: "center" }}>
            Placeholder: Gráfico 3 (ex.: Correlação entre sintomas)
          </div>
        </div>

        <section style={{ marginTop: 20 }}>
          <h2>Insights (exemplo)</h2>
          <ul>
            <li>Maior incidência de febre em faixas etárias X–Y.</li>
            <li>Aumento de suspeitas em períodos após chuvas intensas.</li>
            <li>Certos sintomas aparecem frequentemente em conjunto.</li>
          </ul>
        </section>

        <div style={{ marginTop: 20 }}>
          <Link to="/" className="secondary">
            Voltar para Home
          </Link>
        </div>
      </section>
    </main>
  );
}

export default Graphics;
