# ultrasonic_analysis.py
# Sistema de Análise de Caracterização Ultrassônica
# Autor: [Seu Nome]
# Data: 2026


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import streamlit as st

class UltrasonicAnalyzer:
    """
    Classe para análise de dados ultrassônicos em laminados compósitos
    """
    
    def __init__(self, thickness_mm, density_kgm3, temperature_C=20):
        """
        Inicializa o analisador
        
        Parâmetros:
        -----------
        thickness_mm : float
            Espessura do laminado em milímetros
        density_kgm3 : float
            Densidade em kg/m³
        temperature_C : float
            Temperatura durante medições em °C
        """
        self.h = thickness_mm / 1000  # Converter para metros
        self.rho = density_kgm3
        self.T = temperature_C
        self.measurements = {}
        self.results = {}
        
    def add_measurements(self, direction, delta_t_L, delta_t_T, label='0deg'):
        """
        Adiciona medições de tempo de voo
        
        Parâmetros:
        -----------
        direction : str
            Direção da medição ('0deg', '90deg', 'Z')
        delta_t_L : array-like
            Tempos de voo para ondas longitudinais (μs)
        delta_t_T : array-like
            Tempos de voo para ondas transversais (μs)
        label : str
            Rótulo para identificação
        """
        # Converter para arrays numpy
        dt_L = np.array(delta_t_L) * 1e-6  # Converter μs → s
        dt_T = np.array(delta_t_T) * 1e-6
        
        # Calcular velocidades
        V_L = 2 * self.h / dt_L
        V_T = 2 * self.h / dt_T
        
        # Armazenar
        self.measurements[direction] = {
            'delta_t_L': dt_L,
            'delta_t_T': dt_T,
            'V_L': V_L,
            'V_T': V_T,
            'label': label
        }
        
        print(f"✓ Medições adicionadas para direção {direction}")
        print(f"  {len(dt_L)} medições de ondas L e T")
        
    def remove_outliers(self, data, threshold=2.0):
        """
        Remove outliers usando critério de Z-score
        
        Parâmetros:
        -----------
        data : array-like
            Dados a filtrar
        threshold : float
            Número de desvios padrão para considerar outlier
            
        Retorna:
        --------
        filtered_data : array
            Dados sem outliers
        mask : array (bool)
            Máscara indicando quais pontos foram mantidos
        """
        z_scores = np.abs(stats.zscore(data))
        mask = z_scores < threshold
        
        n_outliers = np.sum(~mask)
        if n_outliers > 0:
            print(f"  ⚠ {n_outliers} outlier(s) removido(s)")
        
        return data[mask], mask
    
    def calculate_statistics(self, direction, remove_outliers=True):
        """
        Calcula estatísticas das medições
        
        Parâmetros:
        -----------
        direction : str
            Direção para análise
        remove_outliers : bool
            Se True, remove outliers antes de calcular
            
        Retorna:
        --------
        stats_dict : dict
            Dicionário com estatísticas
        """
        data = self.measurements[direction]
        
        # Processar ondas longitudinais
        V_L = data['V_L']
        if remove_outliers:
            V_L, mask_L = self.remove_outliers(V_L)
        
        stats_L = {
            'mean': np.mean(V_L),
            'std': np.std(V_L, ddof=1),
            'se': stats.sem(V_L),
            'ci_95': 1.96 * stats.sem(V_L),
            'cv_percent': (np.std(V_L, ddof=1) / np.mean(V_L)) * 100,
            'n': len(V_L),
            'min': np.min(V_L),
            'max': np.max(V_L)
        }
        
        # Processar ondas transversais
        V_T = data['V_T']
        if remove_outliers:
            V_T, mask_T = self.remove_outliers(V_T)
        
        stats_T = {
            'mean': np.mean(V_T),
            'std': np.std(V_T, ddof=1),
            'se': stats.sem(V_T),
            'ci_95': 1.96 * stats.sem(V_T),
            'cv_percent': (np.std(V_T, ddof=1) / np.mean(V_T)) * 100,
            'n': len(V_T),
            'min': np.min(V_T),
            'max': np.max(V_T)
        }
        
        return {'V_L': stats_L, 'V_T': stats_T}
    
    def calculate_elastic_constants(self, nu_12=0.30, nu_13=0.30, nu_23=0.35):
        """
        Calcula constantes elásticas
        
        Parâmetros:
        -----------
        nu_12, nu_13, nu_23 : float
            Coeficientes de Poisson (estimados ou medidos)
        """
        # Obter velocidades médias
        stats_0 = self.calculate_statistics('0deg')
        stats_90 = self.calculate_statistics('90deg')
        stats_Z = self.calculate_statistics('Z')
        
        V_L1 = stats_0['V_L']['mean']
        V_T1 = stats_0['V_T']['mean']
        V_L2 = stats_90['V_L']['mean']
        V_T2 = stats_90['V_T']['mean']
        V_L3 = stats_Z['V_L']['mean']
        V_T3 = stats_Z['V_T']['mean']
        
        # Constantes elásticas (GPa)
        C11 = self.rho * V_L1**2 / 1e9
        C22 = self.rho * V_L2**2 / 1e9
        C33 = self.rho * V_L3**2 / 1e9
        C44 = self.rho * V_T3**2 / 1e9  # G23
        C55 = self.rho * V_T2**2 / 1e9  # G13
        C66 = self.rho * V_T1**2 / 1e9  # G12
        
        # Calcular ν₂₁ (relação de reciprocidade)
        # E₂/E₁ ≈ (V_L2/V_L1)²
        ratio_E = (V_L2/V_L1)**2
        nu_21 = nu_12 * ratio_E
        
        # Módulos de engenharia (GPa)
        # Aproximação para material ortotrópico
        E1 = C11 * (1 - nu_12 * nu_21)
        E2 = C22 * (1 - nu_12 * nu_21)
        E3 = C33
        G12 = C66
        G13 = C55
        G23 = C44
        
        # Propagação de erros
        dV_L1 = stats_0['V_L']['se']
        drho = 5  # Assumindo incerteza de 5 kg/m³
        
        dE1 = E1 * np.sqrt((drho/self.rho)**2 + 2*(dV_L1/V_L1)**2)
        
        # Armazenar resultados
        self.results = {
            'C11': C11, 'C22': C22, 'C33': C33,
            'C44': C44, 'C55': C55, 'C66': C66,
            'E1': E1, 'E2': E2, 'E3': E3,
            'G12': G12, 'G13': G13, 'G23': G23,
            'nu_12': nu_12, 'nu_13': nu_13, 'nu_23': nu_23,
            'dE1': dE1,
            'velocities': {
                '0deg_L': V_L1, '0deg_T': V_T1,
                '90deg_L': V_L2, '90deg_T': V_T2,
                'Z_L': V_L3, 'Z_T': V_T3
            }
        }
        
        print("\n" + "="*60)
        print("RESULTADOS DA CARACTERIZAÇÃO ELÁSTICA")
        print("="*60)
        print(f"\nMÓDULOS DE YOUNG:")
        print(f"  E₁ = {E1:.2f} ± {dE1:.2f} GPa  (direção fibras)")
        print(f"  E₂ = {E2:.2f} GPa  (perpendicular)")
        print(f"  E₃ = {E3:.2f} GPa  (espessura)")
        print(f"\nMÓDULOS DE CISALHAMENTO:")
        print(f"  G₁₂ = {G12:.2f} GPa")
        print(f"  G₁₃ = {G13:.2f} GPa")
        print(f"  G₂₃ = {G23:.2f} GPa")
        print(f"\nÍNDICE DE ANISOTROPIA:")
        print(f"  E₁/E₂ = {E1/E2:.2f}")
        print(f"  V_L(0°)/V_L(90°) = {V_L1/V_L2:.2f}")
        print("="*60 + "\n")
        
        return self.results
    
    def generate_report(self, filename='relatorio_ultrassom.txt'):
        """
        Gera relatório completo em arquivo texto
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RELATÓRIO DE CARACTERIZAÇÃO ULTRASSÔNICA\n")
            f.write("Laminado de Fibra de Carbono\n")
            f.write("="*70 + "\n\n")
            
            f.write("PARÂMETROS DO MATERIAL:\n")
            f.write(f"  Espessura: {self.h*1000:.2f} mm\n")
            f.write(f"  Densidade: {self.rho:.0f} kg/m³\n")
            f.write(f"  Temperatura: {self.T:.1f} °C\n\n")
            

            f.write("VELOCIDADES MÉDIAS:\n")
            # Incluir todas as direções presentes nas medições
            for direction in self.measurements.keys():
                stats = self.calculate_statistics(direction)
                f.write(f"\n  Direção {direction}:\n")
                f.write(f"    V_L = {stats['V_L']['mean']:.1f} ± {stats['V_L']['ci_95']:.1f} m/s\n")
                f.write(f"    V_T = {stats['V_T']['mean']:.1f} ± {stats['V_T']['ci_95']:.1f} m/s\n")
                f.write(f"    CV_L = {stats['V_L']['cv_percent']:.2f}% | CV_T = {stats['V_T']['cv_percent']:.2f}%\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("PROPRIEDADES ELÁSTICAS:\n\n")
            
            r = self.results
            f.write(f"  E₁ = {r['E1']:.2f} ± {r['dE1']:.2f} GPa\n")
            f.write(f"  E₂ = {r['E2']:.2f} GPa\n")
            f.write(f"  E₃ = {r['E3']:.2f} GPa\n")
            f.write(f"  G₁₂ = {r['G12']:.2f} GPa\n")
            f.write(f"  G₁₃ = {r['G13']:.2f} GPa\n")
            f.write(f"  G₂₃ = {r['G23']:.2f} GPa\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✓ Relatório salvo em: {filename}")
    
    def plot_results(self):
        """
        Gera gráficos de visualização
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico 1: Distribuição de velocidades
        ax1 = axes[0, 0]
        directions_dist = ['0deg', '90deg', 'Z']
        if '45deg' in self.measurements:
            directions_dist.insert(2, '45deg')
        for direction in directions_dist:
            data = self.measurements[direction]
            ax1.scatter([direction]*len(data['V_L']), data['V_L'], 
                       alpha=0.6, s=50, label=f'{direction} (L)')
        ax1.set_ylabel('Velocidade (m/s)', fontsize=12)
        ax1.set_title('Distribuição de Velocidades Longitudinais', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Anisotropia
        ax2 = axes[0, 1]
        directions = ['0deg', '90deg', '45deg', 'Z']  # Adicione '45°'
        V_means = [self.calculate_statistics(d)['V_L']['mean'] for d in directions]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax2.bar(directions, V_means, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Velocidade V_L (m/s)', fontsize=12)
        ax2.set_title('Anisotropia no Plano', fontsize=14, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, val in zip(bars, V_means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f} m/s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Gráfico 3: Módulos elásticos
        ax3 = axes[1, 0]
        properties = ['E₁', 'E₂', 'E₃']
        values = [self.results['E1'], self.results['E2'], self.results['E3']]
        bars = ax3.barh(properties, values, color=['#3498db', '#9b59b6', '#f39c12'], 
                       alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Módulo de Young (GPa)', fontsize=12)
        ax3.set_title('Módulos Elásticos', fontsize=14, fontweight='bold')
        ax3.grid(True, axis='x', alpha=0.3)
        
        # Gráfico 4: Controle de qualidade (CV)
        ax4 = axes[1, 1]
        cv_values = []
        labels = []
        directions_cv = ['0deg', '90deg', 'Z']
        if '45deg' in self.measurements:
            directions_cv.insert(2, '45deg')
        for direction in directions_cv:
            stats = self.calculate_statistics(direction)
            cv_values.append(stats['V_L']['cv_percent'])
            labels.append(f'{direction}\nV_L')
        colors_cv = ['green' if cv < 1 else 'orange' if cv < 2 else 'red' for cv in cv_values]
        bars = ax4.bar(labels, cv_values, color=colors_cv, alpha=0.7, edgecolor='black')
        ax4.axhline(y=1, color='green', linestyle='--', label='Excelente (<1%)')
        ax4.axhline(y=2, color='orange', linestyle='--', label='Aceitável (<2%)')
        ax4.set_ylabel('Coeficiente de Variação (%)', fontsize=12)
        ax4.set_title('Controle de Qualidade das Medições', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analise_ultrassom.png', dpi=300, bbox_inches='tight')
        print("✓ Gráficos salvos em: analise_ultrassom.png")
        plt.show()


# ============================================================================
# EXEMPLO DE USO
# ============================================================================


# ================= STREAMLIT APP =====================
def main():
    st.set_page_config(page_title="Análise Ultrassônica", layout="wide")
    st.title("Sistema de Análise Ultrassônica - Laminados de Fibra de Carbono")
    st.markdown("---")


    st.sidebar.header("Parâmetros do Material")
    st.sidebar.markdown("<span style='color:gray'><b>Referência:</b> Espessura: 4.87 mm | Densidade: 1573 kg/m³ | Temperatura: 21.0 °C</span>", unsafe_allow_html=True)
    thickness_mm = st.sidebar.number_input("Espessura (mm)", min_value=0.1, value=4.87, step=0.01, help="Valor de referência: 4.87 mm")
    density_kgm3 = st.sidebar.number_input("Densidade (kg/m³)", min_value=500, value=1573, step=1, help="Valor de referência: 1573 kg/m³")
    temperature_C = st.sidebar.number_input("Temperatura (°C)", min_value=0, value=21, step=1, help="Valor de referência: 21.0 °C")

    st.sidebar.header("Coeficientes de Poisson")
    nu_12 = st.sidebar.number_input("ν₁₂", min_value=0.0, max_value=0.5, value=0.30, step=0.01)
    nu_13 = st.sidebar.number_input("ν₁₃", min_value=0.0, max_value=0.5, value=0.30, step=0.01)
    nu_23 = st.sidebar.number_input("ν₂₃", min_value=0.0, max_value=0.5, value=0.35, step=0.01)

    st.header("Entradas de Medições Ultrassônicas")
    st.write("Insira os tempos de voo (μs) para cada direção e tipo de onda:")

    def input_list(label, default):
        return st.text_area(label, value=", ".join(str(x) for x in default)).replace("\n", ",").replace(";", ",")


    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("Direção 0°")
        delta_t_L_0deg = [float(x) for x in input_list("Ondas Longitudinais (L)", [2.784,2.789,2.786,2.776,2.781,2.779,2.788,2.785,2.787]).split(",") if x.strip()]
        delta_t_T_0deg = [float(x) for x in input_list("Ondas Transversais (T)", [5.273,5.281,5.275,5.265,5.270,5.268,5.279,5.276,5.278]).split(",") if x.strip()]
    with col2:
        st.subheader("Direção 45°")
        delta_t_L_45deg = [float(x) for x in input_list("Ondas Longitudinais (L)", [4.248,4.251,4.250,4.247,4.249,4.250,4.252,4.248,4.251]).split(",") if x.strip()]
        delta_t_T_45deg = [float(x) for x in input_list("Ondas Transversais (T)", [6.523,6.528,6.525,6.521,6.524,6.525,6.529,6.522,6.527]).split(",") if x.strip()]
    with col3:
        st.subheader("Direção 90°")
        delta_t_L_90deg = [float(x) for x in input_list("Ondas Longitudinais (L)", [4.870,4.875,4.873,4.868,4.872,4.871,4.874,4.869,4.873]).split(",") if x.strip()]
        delta_t_T_90deg = [float(x) for x in input_list("Ondas Transversais (T)", [6.845,6.852,6.848,6.843,6.847,6.846,6.850,6.844,6.848]).split(",") if x.strip()]
    with col4:
        st.subheader("Direção Z (espessura)")
        delta_t_L_Z = [float(x) for x in input_list("Ondas Longitudinais (L)", [3.248,3.251,3.250,3.247,3.249,3.250,3.252,3.248,3.251]).split(",") if x.strip()]
        delta_t_T_Z = [float(x) for x in input_list("Ondas Transversais (T)", [6.123,6.128,6.125,6.121,6.124,6.125,6.129,6.122,6.127]).split(",") if x.strip()]


    if st.button("Processar Análise Ultrassônica"):
        analyzer = UltrasonicAnalyzer(
            thickness_mm=thickness_mm,
            density_kgm3=density_kgm3,
            temperature_C=temperature_C
        )
        analyzer.add_measurements('0deg', delta_t_L_0deg, delta_t_T_0deg)
        if delta_t_L_45deg and delta_t_T_45deg:
            analyzer.add_measurements('45deg', delta_t_L_45deg, delta_t_T_45deg)
        analyzer.add_measurements('90deg', delta_t_L_90deg, delta_t_T_90deg)
        analyzer.add_measurements('Z', delta_t_L_Z, delta_t_T_Z)
        results = analyzer.calculate_elastic_constants(nu_12=nu_12, nu_13=nu_13, nu_23=nu_23)

        st.success("Análise concluída!")


        st.subheader("Resultados Principais")
        st.write(f"**Módulo de Young E₁:** {results['E1']:.2f} ± {results['dE1']:.2f} GPa")
        st.write(f"**Módulo de Young E₂:** {results['E2']:.2f} GPa")
        st.write(f"**Módulo de Young E₃:** {results['E3']:.2f} GPa")
        st.write(f"**Módulo de Cisalhamento G₁₂:** {results['G12']:.2f} GPa")
        st.write(f"**Módulo de Cisalhamento G₁₃:** {results['G13']:.2f} GPa")
        st.write(f"**Módulo de Cisalhamento G₂₃:** {results['G23']:.2f} GPa")
        st.write(f"**Índice de Anisotropia (E₁/E₂):** {results['E1']/results['E2']:.2f}")


        # Resultados estatísticos para todas as direções presentes
        st.markdown('---')
        st.subheader('Resultados Estatísticos por Direção')
        stats_table = []
        for direction in analyzer.measurements.keys():
            stats = analyzer.calculate_statistics(direction)
            stats_table.append({
                'Direção': direction,
                'V_L Média (m/s)': f"{stats['V_L']['mean']:.2f}",
                'V_L IC95 (m/s)': f"±{stats['V_L']['ci_95']:.2f}",
                'V_L CV (%)': f"{stats['V_L']['cv_percent']:.2f}",
                'V_T Média (m/s)': f"{stats['V_T']['mean']:.2f}",
                'V_T IC95 (m/s)': f"±{stats['V_T']['ci_95']:.2f}",
                'V_T CV (%)': f"{stats['V_T']['cv_percent']:.2f}"
            })
        st.dataframe(stats_table, hide_index=True)

        # Gráfico comparativo das médias e IC95
        import matplotlib.pyplot as plt
        import numpy as np
        fig_stats, ax_stats = plt.subplots(figsize=(8,5))
        directions = list(analyzer.measurements.keys())
        means_VL = [analyzer.calculate_statistics(d)['V_L']['mean'] for d in directions]
        ci95_VL = [analyzer.calculate_statistics(d)['V_L']['ci_95'] for d in directions]
        means_VT = [analyzer.calculate_statistics(d)['V_T']['mean'] for d in directions]
        ci95_VT = [analyzer.calculate_statistics(d)['V_T']['ci_95'] for d in directions]
        x = np.arange(len(directions))
        width = 0.35
        ax_stats.bar(x - width/2, means_VL, width, yerr=ci95_VL, capsize=6, label='V_L', color='#3498db')
        ax_stats.bar(x + width/2, means_VT, width, yerr=ci95_VT, capsize=6, label='V_T', color='#e67e22')
        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels(directions)
        ax_stats.set_ylabel('Velocidade (m/s)')
        ax_stats.set_title('Comparativo das Médias e IC95 das Velocidades por Direção')
        ax_stats.legend()
        st.pyplot(fig_stats)

        st.markdown("---")
        st.subheader("Gráficos de Resultados")
        # Gráficos
        import io
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12,8))
        analyzer.plot_results()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        st.image(buf)

        st.markdown("---")
        st.subheader("Download do Relatório")
        analyzer.generate_report('relatorio_caracterizacao.txt')
        with open('relatorio_caracterizacao.txt', 'rb') as f:
            st.download_button("Baixar Relatório TXT", f, file_name='relatorio_caracterizacao.txt')

if __name__ == "__main__":
    main()