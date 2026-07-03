type BoxProps = {
  x: number;
  y: number;
  w: number;
  h: number;
  l1: string;
  l2?: string;
  variant?: "step" | "highlight";
};

function Box({ x, y, w, h, l1, l2, variant = "step" }: BoxProps) {
  const cx = x + w / 2;
  const cy = y + h / 2;
  const fill = variant === "highlight" ? "#0f766e" : "#ffffff";
  const text = variant === "highlight" ? "#ffffff" : "#1f2937";
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={12}
        fill={fill}
        stroke="#0f766e"
        strokeWidth={1.6}
      />
      <text
        x={cx}
        y={l2 ? cy - 8 : cy}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={13}
        fontWeight={700}
        fill={text}
      >
        {l1}
      </text>
      {l2 && (
        <text
          x={cx}
          y={cy + 9}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={13}
          fontWeight={700}
          fill={text}
        >
          {l2}
        </text>
      )}
    </g>
  );
}

function Cylinder({
  x,
  y,
  w,
  h,
  l1,
  l2,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  l1: string;
  l2?: string;
}) {
  const rx = w / 2;
  const ry = 6;
  const cx = x + w / 2;
  const lidBottom = y + 2 * ry;
  const bodyBottom = y + ry + h;
  const textCenter = (lidBottom + bodyBottom) / 2;
  const body = `M${x} ${y + ry} a ${rx} ${ry} 0 0 0 ${w} 0 l 0 ${h} a ${rx} ${ry} 0 0 1 ${-w} 0 z`;
  const lid = `M${x} ${y + ry} a ${rx} ${ry} 0 0 0 ${w} 0`;
  return (
    <g>
      <path d={body} fill="#f1f5f9" stroke="#64748b" strokeWidth={1.4} />
      <path d={lid} fill="none" stroke="#64748b" strokeWidth={1.4} />
      <text
        x={cx}
        y={l2 ? textCenter - 8 : textCenter}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={12.5}
        fontWeight={700}
        fill="#334155"
      >
        {l1}
      </text>
      {l2 && (
        <text
          x={cx}
          y={textCenter + 8}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={12.5}
          fontWeight={700}
          fill="#334155"
        >
          {l2}
        </text>
      )}
    </g>
  );
}

const etapas = [
  {
    num: "01",
    titulo: "Dados do SINAN (2014 a 2021)",
    texto:
      "Os dados vêm do SINAN, o sistema de notificação de doenças do Ministério da Saúde, que fica aberto no DataSUS. Usamos os registros de 2014 a 2021.",
  },
  {
    num: "02",
    titulo: "Leitura e conferência",
    texto:
      "Os arquivos são enormes, então lemos aos poucos e conferimos ano a ano quantos casos entraram, quantos foram descartados e o que ficou faltando.",
  },
  {
    num: "03",
    titulo: "Preparação e variáveis",
    texto:
      "Os dados são limpos e viram as informações que o modelo usa: idade, sintomas, local. A que mais fez diferença foi o quanto a dengue vinha se confirmando naquela cidade nas semanas anteriores.",
  },
  {
    num: "04",
    titulo: "Treino (2017 a 2019)",
    texto:
      "Três modelos diferentes aprendem com os casos de 2017 a 2019 e depois são combinados numa resposta só. Os ajustes finos foram feitos olhando para 2020.",
  },
  {
    num: "05",
    titulo: "Teste em 2021",
    texto:
      "A avaliação de verdade usa os casos de 2021, um ano que o modelo nunca tinha visto. Assim dá pra saber como ele se sai com casos novos, e não com dados que já conhecia.",
  },
  {
    num: "06",
    titulo: "Modelos salvos",
    texto:
      "Depois de prontos, os modelos e os arquivos que eles precisam ficam guardados juntos, prontos para uso.",
  },
  {
    num: "07",
    titulo: "API",
    texto:
      "Uma API carrega esses modelos e prepara os dados de cada pessoa do mesmo jeito que foi feito no treino, para a resposta bater com o que foi testado.",
  },
  {
    num: "08",
    titulo: "O site",
    texto:
      "É o que você usa aqui: preencher um caso na triagem, ou sortear um caso real de 2021 na simulação, e ver o que os modelos preveem.",
  },
];

function Pipeline() {
  return (
    <main className="container">
      <section className="card">
        <span className="analise-eyebrow">Como funciona</span>
        <h1 className="pipeline-titulo">Pipeline de dados do projeto</h1>
        <p className="pipeline-intro">
          Aqui dá pra ver o caminho que os dados fazem neste projeto, do banco do
          SINAN até a resposta que aparece no site. De um lado ficam as etapas de
          preparar os dados e treinar os modelos; do outro, a parte que já está no
          ar respondendo às previsões.
        </p>

        <div className="pipeline-diagrama">
          <svg viewBox="0 0 900 372" role="img" aria-label="Diagrama do pipeline">
            <defs>
              <marker
                id="seta"
                markerWidth="9"
                markerHeight="9"
                refX="7"
                refY="3"
                orient="auto"
                markerUnits="strokeWidth"
              >
                <path d="M0,0 L7,3 L0,6 Z" fill="#64748b" />
              </marker>
            </defs>

            {/* Divisor experimentacao / producao */}
            <line
              x1="20"
              y1="262"
              x2="884"
              y2="262"
              stroke="#94a3b8"
              strokeWidth="1.2"
              strokeDasharray="6 6"
            />
            <text x="24" y="250" fontSize="11" fontWeight="700" fill="#0f766e">
              experimentação e teste
            </text>
            <text x="24" y="284" fontSize="11" fontWeight="700" fill="#b45309">
              produção e serving
            </text>

            {/* Container tracejado das etapas manuais */}
            <rect
              x="170"
              y="124"
              width="578"
              height="112"
              rx="14"
              fill="none"
              stroke="#94a3b8"
              strokeWidth="1.2"
              strokeDasharray="6 6"
            />
            <text x="186" y="145" fontSize="11.5" fontWeight="700" fill="#64748b">
              Etapas de experimentação (executadas manualmente)
            </text>

            {/* Setas: fluxo de experimentacao (esquerda para direita) */}
            <g stroke="#64748b" strokeWidth="1.6" fill="none" markerEnd="url(#seta)">
              <path d="M86 74 L86 160" />
              <path d="M146 189 L190 189" />
              <path d="M340 189 L356 189" />
              <path d="M516 189 L532 189" />
              <path d="M728 189 L764 189" />
              {/* Modelo treinado desce para o registro */}
              <path d="M819 218 L819 300" />
              {/* Fluxo de producao (direita para esquerda) */}
              <path d="M753 334 L536 334" />
              <path d="M356 329 L176 329" />
            </g>

            {/* Fonte de dados */}
            <Cylinder x={20} y={12} w={132} h={48} l1="Dados SINAN" l2="2014-2021" />

            {/* Linha de experimentacao */}
            <Box x={26} y={160} w={120} h={58} l1="Extração e" l2="análise" />
            <Box x={190} y={160} w={150} h={58} l1="Preparação" l2="e features" />
            <Box x={356} y={160} w={160} h={58} l1="Treino" l2="2017-2019" />
            <Box x={532} y={160} w={196} h={58} l1="Avaliação" l2="temporal 2021" />
            <Box x={764} y={160} w={110} h={58} l1="Modelo" l2="treinado" />

            {/* Linha de producao */}
            <Cylinder x={753} y={298} w={132} h={48} l1="Modelos" l2="salvos" />
            <Box x={356} y={300} w={180} h={58} l1="Serviço do" l2="modelo (API)" />
            <Box
              x={26}
              y={300}
              w={150}
              h={58}
              l1="Predição"
              l2="no site"
              variant="highlight"
            />
          </svg>
        </div>

        <div className="pipeline-etapas">
          {etapas.map((etapa) => (
            <div className="pipeline-etapa" key={etapa.num}>
              <span className="pipeline-etapa-num">{etapa.num}</span>
              <h3>{etapa.titulo}</h3>
              <p>{etapa.texto}</p>
            </div>
          ))}
        </div>

        <div className="pipeline-destaque">
          <strong>Por que treinar e testar em anos diferentes?</strong>
          <p>
            Uma forma comum de testar é misturar todos os anos e separar treino e
            teste ao acaso. O problema é que aí o modelo acaba treinando com dados
            do mesmo período em que é avaliado, e o resultado parece melhor do que
            é na prática. A gente preferiu treinar com os anos mais antigos e
            testar num ano mais recente, separado. O número fica mais baixo, mas é
            o que o modelo realmente entrega quando aparece um caso novo.
          </p>
        </div>
      </section>
    </main>
  );
}

export default Pipeline;
