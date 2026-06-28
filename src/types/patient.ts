export type PatientData = {
  ageYears: string;
  sex: string;
  pregnancyStatus: string;
  race: string;
  educationLevel: string;
  occupationCode: string;
  occupationName: string;        // label visível; código fica em occupationCode
  residenceState: string;        // código IBGE da UF (guardado internamente)
  residenceStateLabel: string;   // sigla da UF (ex: "RJ")
  residenceMunicipality: string; // código IBGE do município
  residenceHealthRegion: string; // código da região de saúde
  notificationDate: string;
  symptomOnsetDate: string;
  daysToNotification: string;    // calculado automaticamente
  symptomEpiWeekNumber: string;  // calculado automaticamente
  symptomEpiYear: string;        // calculado automaticamente
};