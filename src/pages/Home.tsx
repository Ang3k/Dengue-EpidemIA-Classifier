import PredictionSimulator from "../components/PredictionSimulator";

function Home() {
  return (
    <main className="container">
      <section className="card">
        <h1>Dengue Sense Classifier</h1>

        <p>
          O Dengue Sense Classifier é a interface de um projeto de aprendizado de
          máquina aplicado à epidemiologia da dengue. A partir de registros
          oficiais do SINAN, treinamos um modelo que aprende a distinguir casos{" "}
          <strong>confirmados</strong> de <strong>descartados</strong>. Com isso,
          ele ajuda na triagem de suspeitas, estimando a partir dos sintomas e dos
          dados do paciente o quão provável é que se trate de um caso real de
          dengue.
        </p>

        <div className="home-section">
          <h2>Como funciona</h2>
          <p>
            Na página de <strong>Triagem</strong>, você preenche os dados
            principais do paciente (idade, sexo, gestação, entre outros) e marca
            os sintomas e achados clínicos observados. O sistema combina essas
            informações e devolve uma probabilidade de o caso ser dengue, junto
            com uma orientação.
          </p>
          <p>
            Já a página de <strong>Panorama Epidemiológico</strong> reúne os
            gráficos históricos da análise original de 2017 a 2019,
            como a sazonalidade, os sintomas mais discriminantes e os perfis por
            sexo e ocupação.
          </p>
          <p>
            <em>
              A triagem envia os dados preenchidos aos modelos treinados. Na
              simulação abaixo, o sistema seleciona um registro histórico
              anonimizado do conjunto de teste e compara sua classificação com
              as probabilidades calculadas pelos modelos.
            </em>
          </p>
        </div>

        <div className="home-section">
          <h2>Sobre os dados</h2>
          <p>
            O modelo usa registros oficiais de dengue notificados no Brasil pelo
            SINAN entre <strong>2014 e 2021</strong>: 11,4 milhões de notificações
            brutas e quase 10 milhões de casos rotulados. Cada registro traz sintomas, dados
            demográficos (idade, sexo, raça/cor, escolaridade, ocupação), a
            localização de residência e as datas de notificação e de início dos
            sintomas. O que o modelo aprende a prever é a classificação final de
            cada caso: confirmado ou descartado.
          </p>
        </div>

        <div className="home-section">
          <h2>Sobre o modelo</h2>
          <p>
            O modelo responde a uma pergunta epidemiológica: com o que se sabe de
            um caso na hora da notificação, qual a chance de ele ser dengue
            confirmada? Para isso, transformamos os dados brutos em variáveis
            úteis, como os sintomas em forma binária e suas combinações, a
            sazonalidade do ano em forma cíclica (a dengue tem pico no verão e no
            outono), o perfil demográfico e a região de residência, já que a
            endemicidade muda bastante pelo país. Comparamos uma rede neural
            MLP com embeddings, XGBoost e LightGBM.
          </p>
          <p>
            Também tomamos cuidado para o modelo não trapacear. Nenhuma
            informação que só existe depois que o caso é encerrado entra como
            variável, e o ano em si fica de fora, para que ele aprenda padrões que
            se repetem em vez de decorar um ano específico. Como em vigilância é
            pior deixar passar um caso real do que dar um alarme falso, o ponto de
            decisão pode ser ajustado para errar menos os casos verdadeiros.
          </p>
        </div>

        <PredictionSimulator />

        <p className="home-aviso">
          Esta ferramenta tem caráter informativo e de apoio. Ela não substitui
          a avaliação de um profissional de saúde, que deve sempre ser
          procurado diante de qualquer suspeita de dengue.
        </p>
      </section>
    </main>
  );
}

export default Home;
