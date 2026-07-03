import casosPorMes from "../../reports/figures/casos_por_mes.png";
import sintomas from "../../reports/figures/sintomas_confirmados_vs_descartados.png";
import casosPorSexo from "../../reports/figures/casos_confirmados_por_sexo.png";
import ocupacoesFeminino from "../../reports/figures/ocupacoes_confirmadas_feminino.png";
import ocupacoesMasculino from "../../reports/figures/ocupacoes_confirmadas_masculino.png";

type Insight = { titulo: string; texto: string };
type Imagem = { src: string; alt: string; legenda?: string };

type Analise = {
  numero: string;
  chamada: string;
  imagens: Imagem[];
  insights: Insight[];
};

const analises: Analise[] = [
  {
    numero: "01",
    chamada: "A dengue tem pico claro no outono. O modelo precisa capturar isso.",
    imagens: [
      {
        src: casosPorMes,
        alt: "Gráfico de barras com o total de casos de dengue por mês entre 2017 e 2019",
        legenda: "Total de casos de dengue por mês, 2017 a 2019",
      },
    ],
    insights: [
      {
        titulo: "Pico absoluto",
        texto:
          "Maio concentra 620 mil casos, o maior volume de todo o período. Abril (578 mil) e Março (396 mil) completam o pico do outono, que responde por 60% das notificações anuais.",
      },
      {
        titulo: "Escalada no verão",
        texto:
          "Janeiro e Fevereiro já sinalizam o surto (177 e 258 mil), indicando que a transmissão começa antes do pico do calor.",
      },
      {
        titulo: "Vale no inverno/primavera",
        texto:
          "Queda expressiva de agosto a outubro (69 a 85 mil casos), mas nunca chega a zero: o Aedes mantém atividade o ano todo.",
      },
    ],
  },
  {
    numero: "02",
    chamada:
      "Todos os sintomas são mais frequentes em confirmados, mas nenhum discrimina sozinho.",
    imagens: [
      {
        src: sintomas,
        alt: "Gráfico comparando a frequência dos sintomas entre casos confirmados e descartados",
        legenda: "Sintomas mais frequentes por classificação",
      },
    ],
    insights: [
      {
        titulo: "Tríade dominante",
        texto:
          "Febre (85,9%), cefaleia (80,2%) e mialgia (79,6%) passam de 79% nos confirmados: são quase universais e pouco discriminantes quando vistos isolados.",
      },
      {
        titulo: "Maior diferença proporcional",
        texto:
          "Exantema tem o maior gap relativo entre as classes: 25,4% nos confirmados vs 15,6% nos descartados (+63% relativo). Um dos sinais mais diferenciadores.",
      },
      {
        titulo: "Dor retro-orbital",
        texto:
          "37,6% nos confirmados vs 26,8% nos descartados, uma diferença expressiva e consistente com a clínica clássica da dengue.",
      },
    ],
  },
  {
    numero: "03",
    chamada: "Mulheres representam 55,7% dos casos confirmados.",
    imagens: [
      {
        src: casosPorSexo,
        alt: "Gráfico da distribuição dos casos confirmados de dengue por sexo",
        legenda: "Distribuição dos casos confirmados por sexo",
      },
    ],
    insights: [
      {
        titulo: "Distribuição",
        texto:
          "933.052 casos femininos (55,7%) contra 743.044 masculinos (44,3%), uma diferença de cerca de 190 mil casos em 3 anos.",
      },
      {
        titulo: "Hipótese de exposição",
        texto:
          "Segundo o Estado de Minas, a maior proporção feminina pode estar ligada à exposição doméstica. O Aedes aegypti se reproduz principalmente em reservatórios de água em casa, então quem passa mais tempo em casa fica mais exposto.",
      },
      {
        titulo: "Interação com ocupação",
        texto:
          "“Dona de casa” lidera entre as mulheres (77.874 casos), reforçando a hipótese de que o ambiente doméstico é o principal vetor de exposição feminina.",
      },
    ],
  },
  {
    numero: "04",
    chamada: "O perfil de ocupação revela padrões de exposição distintos por sexo.",
    imagens: [
      {
        src: ocupacoesFeminino,
        alt: "Ocupações com maior registro de casos confirmados entre mulheres",
        legenda: "Ocupações (Feminino)",
      },
      {
        src: ocupacoesMasculino,
        alt: "Ocupações com maior registro de casos confirmados entre homens",
        legenda: "Ocupações (Masculino)",
      },
    ],
    insights: [
      {
        titulo: "Exposição doméstica",
        texto: "Dona de casa lidera entre as mulheres, com 77.874 casos.",
      },
      {
        titulo: "Estudantes em ambos",
        texto:
          "Lidera entre os homens (59 mil) e é a 2ª colocada entre as mulheres (60 mil).",
      },
      {
        titulo: "Exposição externa",
        texto: "Pedreiro e trabalhador agropecuário aparecem só entre os homens.",
      },
    ],
  },
];

function Graphics() {
  return (
    <main className="container">
      <section className="card card-analise">
        <h1>Panorama Epidemiológico da Dengue</h1>

        <p>
          Reunimos aqui as principais descobertas da análise exploratória dos
          casos de dengue notificados no Brasil entre 2017 e 2019. Cada gráfico
          vem acompanhado dos insights que extraímos sobre o comportamento da
          doença.
        </p>

        {analises.map((analise) => (
          <article className="analise" key={analise.numero}>
            <span className="analise-eyebrow">Análise · {analise.numero}</span>
            <h2 className="analise-chamada">{analise.chamada}</h2>

            <div
              className={
                analise.imagens.length > 1
                  ? "analise-figuras analise-figuras-dupla"
                  : "analise-figuras"
              }
            >
              {analise.imagens.map((imagem) => (
                <figure className="analise-figura" key={imagem.src}>
                  <img src={imagem.src} alt={imagem.alt} loading="lazy" />
                  {imagem.legenda && (
                    <figcaption>{imagem.legenda}</figcaption>
                  )}
                </figure>
              ))}
            </div>

            <div className="analise-insights">
              {analise.insights.map((insight) => (
                <div className="insight-card" key={insight.titulo}>
                  <h3>{insight.titulo}</h3>
                  <p>{insight.texto}</p>
                </div>
              ))}
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}

export default Graphics;
