import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Logismart-App")

# Opciones en el menú lateral
option = st.sidebar.selectbox(
    'Selecciona una opción:',
    ('Forecast', 'Stock de Seguridad y Punto de Reorden', 'ABC', 'Ranking')
)
 
# Función para la opción "Hola Mundo"
def hola_mundo():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    st.header('Forecast')

    st.sidebar.header("Cargar archivo de ventas")
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de ventas", type=["csv"])

    # Lista de Familias
    familias = [
        "TERMINAL", "TUERCA", "LLAVE", "CONECTOR", "MAP", "CODO", "COPLA",
        "GOLILLA", "BANDA", "BRIDA", "COLLAR", "UNION", "CABEZAL", "PERNO",
        "TUBO", "EMPAQUETADURA", "CAMARA", "TUBERIA"
    ]

    # Menú desplegable para seleccionar la familia
    familia_seleccionada = st.sidebar.selectbox("Selecciona la familia para el forecast", familias)

    if uploaded_file is not None:
        # Lectura de los archivos subidos
        dataset = pd.read_csv(uploaded_file, index_col=0, encoding='latin-1')

        # Procesamiento de datos (ajustado según tu necesidad)
        df = dataset.drop(columns=[
            'Organización de ventas', 'Pagador', 'Nombre', 'Rut', 'Gr.Clt-CLI',
            'Gr.clientes-CLI', 'Gr.Ven.-CLI', 'Gr. Vendedores - CLI',
            'Gr.Precio-CLI', 'Nº pedido cliente', 'Fe.PedCli', 'Factura',
            'Referencia', 'Clase de factura', 'ZonaVen',
            'Zona de ventas', 'GI', 'Grupo de imputación', 'CPag', 'FeValFijad',
            'Referencia de pago', 'StatusC', 'An.', 'Pos.Fac',
            '% margen s/venta', '% margen s/cost', 'Subtotal 6', 'Mon..5', 'Gr.Cl1',
            'Gr.Clientes 1', 'Grupo art', 'Grupo de artículos', 'GrMat',
            'Grupo materiales', 'GImMat', 'Gr.Imp.Material', 'OfVta',
            'Oficina de ventas', 'Doc.venta', 'Tp.DC', 'Pos.Ventas', 'Gr.Ven.-PED',
            'Gr.vendedores-PED', 'Gr.Cl-PED', 'Gr.clientes-PED', 'GP-PED',
            'Gr.Precio-PED', 'Costes internos posición', 'Mon..6', 'Proc.empres.',
            'CeBe'
        ])
        df['Fecha factura'] = pd.to_datetime(df['Fecha factura'])
        df.rename(columns={'Fecha factura': 'fecha'}, inplace=True)
        df['Año'] = df.fecha.dt.year
        df = df[df['fecha'] >= '2022-01-1']
        df.rename(columns={'Denominación': 'Denominacion', 'Ctd.facturada2': 'Unidad'}, inplace=True)
        df = df[~df['Denominacion'].str.contains('POLIMERO|CLORURO|SULFATO|FLETE|UREA|REACTIVO')]
        df['Cantidad'] = df['Cantidad'].str.replace(",", "").str.replace(".", "")
        df['NETO'] = df['NETO'].str.replace(",", "").str.replace(".", "")
        df['Margen'] = df['Margen'].str.replace(",", "").str.replace(".", "")
        df = df[df['Cantidad'].notna()]
        df['Material'] = df['Material'].astype(str)
        df['Cantidad'] = df['Cantidad'].astype(int)
        df['NETO'] = df['NETO'].astype(int)
        df['Margen'] = df['Margen'].astype(int)
        df = df[~df['Material'].str.startswith('S', na=False)]
        df['Familia'] = df['Denominacion'].str.split().str[0]
        df['date'] = df.fecha.dt.strftime('%Y-%m-%d')
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df.date.dt.month
        df['year'] = df.date.dt.year
        df['week'] = df.date.dt.isocalendar().week

        # Filtrar por la familia seleccionada
        df = df[df['Familia'] == familia_seleccionada]
        CV = ['Cantidad', 'NETO']
        Cant_Venta = st.sidebar.selectbox("Selecciona el tipo de Forecast", CV)
        time_series=df.groupby(['week','month','year']).agg(date=('date','first'), demanda=(Cant_Venta, np.sum)).reset_index().sort_values('date')
        q25, q50, q75 = np.percentile(time_series.demanda, (25,50,75))
        iqr = q75 - q25
        min_grade = q25 - 1.5*iqr
        max_grade = q75 + 1.5*iqr
        
        max_value =time_series.demanda[(time_series.demanda>max_grade) & (time_series.demanda< time_series.demanda.max()) ].mean() 
        time_series.demanda = np.where(time_series.demanda>max_value, max_value,time_series.demanda)
        time_series['date']=pd.to_datetime(time_series['date'])
        time_series= time_series.set_index('date')
        monthly_series = time_series.demanda.resample('M').sum()

        # Definir la serie de tiempo mensual (debes tener esta definida)
        # monthly_series = ...

        def ajustar_y_predecir(serie, trend=None, seasonal=None, seasonal_periods=12):
            modelo = ExponentialSmoothing(serie, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
            ajuste = modelo.fit()
            predicciones = ajuste.fittedvalues
            mape = np.mean(np.abs((serie - predicciones) / serie)) * 100
            return ajuste, predicciones, mape

        # Probar diferentes configuraciones
        configuraciones = [
            {'trend': None, 'seasonal': None},
            {'trend': 'add', 'seasonal': None},
            {'trend': 'mul', 'seasonal': None},
            {'trend': None, 'seasonal': 'add', 'seasonal_periods': 12},
            {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12},
            {'trend': 'mul', 'seasonal': 'add', 'seasonal_periods': 12}
        ]

        # Función para calcular las métricas de error
        def calcular_metricas(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return mae, mse, rmse, mape

        # Variables para almacenar el mejor modelo y su MAPE
        mejor_mape = np.inf
        mejor_modelo = None
        mejores_predicciones = None

        # Ajustar y graficar solo el mejor modelo
        for config in configuraciones:
            ajuste, predicciones, mape = ajustar_y_predecir(monthly_series, **config)
            mae, mse, rmse, _ = calcular_metricas(monthly_series, predicciones)
            
            # Actualizar el mejor modelo basado en MAPE
            if mape < mejor_mape:
                mejor_mape = mape
                mejor_modelo = ajuste
                mejores_predicciones = predicciones

        # Graficar el mejor modelo encontrado
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(monthly_series, label='Datos Originales')
        ax.plot(mejores_predicciones, label='Predicciones')
        ax.set_title(f"Mejor Modelo | MAPE: {mejor_mape:.2f}%")
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Valor')
        ax.legend()
        st.pyplot(fig)

        # Forecast de los próximos 12 meses con el mejor modelo
        forecast = mejor_modelo.forecast(12)
        st.subheader("Forecast de los próximos 12 meses:")
        st.write(round(forecast))
# Función para la opción "Gráfico"
def mostrar_grafico():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import datetime
    import inventorize as inv
    import io
    from io import BytesIO

    st.sidebar.header("Cargar archivo de ventas")
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de ventas", type=["csv"])

    # Carga del archivo de lead time
    st.sidebar.header("Cargar archivo de lead time")
    uploaded_file_ldtime = st.sidebar.file_uploader("Sube tu archivo Excel de lead time", type=["xlsx"])
    
    if uploaded_file is not None and uploaded_file_ldtime is not None:
        # Lectura de los archivos subidos
        dataset = pd.read_csv(uploaded_file, index_col=0, encoding='latin-1')
        ldtime = pd.read_excel(uploaded_file_ldtime)

        # Procesamiento de datos
        df = dataset.drop(columns=[
            'Organización de ventas', 'Pagador', 'Nombre', 'Rut', 'Gr.Clt-CLI',
            'Gr.clientes-CLI', 'Gr.Ven.-CLI', 'Gr. Vendedores - CLI',
            'Gr.Precio-CLI', 'Nº pedido cliente', 'Fe.PedCli', 'Factura',
            'Referencia', 'Clase de factura', 'ZonaVen',
            'Zona de ventas', 'GI', 'Grupo de imputación', 'CPag', 'FeValFijad',
            'Referencia de pago', 'StatusC', 'An.', 'Pos.Fac',
            '% margen s/venta', '% margen s/cost', 'Subtotal 6', 'Mon..5', 'Gr.Cl1',
            'Gr.Clientes 1', 'Grupo art', 'Grupo de artículos', 'GrMat',
            'Grupo materiales', 'GImMat', 'Gr.Imp.Material', 'OfVta',
            'Oficina de ventas', 'Doc.venta', 'Tp.DC', 'Pos.Ventas', 'Gr.Ven.-PED',
            'Gr.vendedores-PED', 'Gr.Cl-PED', 'Gr.clientes-PED', 'GP-PED',
            'Gr.Precio-PED', 'Costes internos posición', 'Mon..6', 'Proc.empres.',
            'CeBe'
        ])
        df['Fecha factura'] = pd.to_datetime(df['Fecha factura'])
        df.rename(columns={'Fecha factura': 'fecha'}, inplace=True)
        df['Año'] = df.fecha.dt.year
        df = df[df['fecha'] >= '2022-01-1']
        df.rename(columns={'Denominación': 'Denominacion', 'Ctd.facturada2': 'Unidad'}, inplace=True)
        df = df[~df['Denominacion'].str.contains('POLIMERO|CLORURO|SULFATO|FLETE|UREA|REACTIVO')]
        df['Cantidad'] = df['Cantidad'].str.replace(",", "").str.replace(".", "")
        df['NETO'] = df['NETO'].str.replace(",", "").str.replace(".", "")
        df['Margen'] = df['Margen'].str.replace(",", "").str.replace(".", "")
        df = df[df['Cantidad'].notna()]
        df['Material'] = df['Material'].astype(str)
        df['Cantidad'] = df['Cantidad'].astype(int)
        df['NETO'] = df['NETO'].astype(int)
        df['Margen'] = df['Margen'].astype(int)
        df = df[~df['Material'].str.startswith('S', na=False)]
        df['Familia'] = df['Denominacion'].str.split().str[0]
        df['date'] = df.fecha.dt.strftime('%Y-%m-%d')
        df['date'] = pd.to_datetime(df['date'])
        abc=df
        
        # Aplicar filtro de fecha
        
        ventas_group = abc.groupby('Denominacion').agg(codigo_material=('Material', np.unique), cantidad_total= ('Cantidad',np.sum),unidad_medida=('UM', np.unique),ventas_totales= ('NETO',np.sum), margen_total =('Margen', np.sum),familia=('Familia', np.unique)).reset_index()
        ventas_group.familia=ventas_group.familia.astype(str)
        ventas_group.unidad_medida=ventas_group.unidad_medida.astype(str)
        ventas_group.codigo_material=ventas_group.codigo_material.astype(str)
        b=inv.productmix(ventas_group['Denominacion'], ventas_group['ventas_totales'],ventas_group['margen_total'])
        b.head(20)
        df_group = df.groupby(['Denominacion', 'fecha']).agg(ventas_totales=('NETO', 'sum')).reset_index()

        cv_data = df_group.groupby('Denominacion').agg(promedio_ventas=('ventas_totales', 'mean'), sd_ventas=('ventas_totales', 'std')).reset_index()
        cv_data['cv_squared'] = (cv_data['sd_ventas'] / cv_data['promedio_ventas']) ** 2

        producto_date = df_group.groupby('Denominacion').agg(cantidad=('Denominacion', 'count')).reset_index()
        skus = producto_date.Denominacion.unique()
        fecha = max(df.date)
        nuevafecha = fecha - datetime.timedelta(days=30 * 4)
        last_four = df[df.fecha > nuevafecha]
        last_four['month'] = last_four.date.dt.month
        last_four['year'] = last_four.date.dt.year
        last_four['week'] = last_four.date.dt.isocalendar().week

        aa = last_four.groupby(['week', 'month', 'year', 'Material']).agg(
            nom_material=('Denominacion', np.unique), date=('date', 'first'), total_semanal=('Cantidad', np.sum), total_ventas=('NETO', np.sum)).reset_index()

        groupedd = aa.groupby('Material').agg(promedio=('total_semanal', np.mean), sd=('total_semanal', 'std'), total_demanda=('total_semanal', np.sum), total_ventas=('total_ventas', np.sum)).reset_index()
        for_abc = inv.productmix(groupedd['Material'], groupedd['total_demanda'], groupedd['total_ventas'])

        mapping = {'A_A': 0.95, "A_C": 0.95, "C_A": 0.8, "A_B": 0.95, 'B_A': 0.7, "B_C": 0.75, "C_C": 0.7, "B_B": 0.7, "C_B": 0.8}
        for_abc['service_level'] = for_abc.product_mix.map(mapping)

        ldtime = ldtime[ldtime['Material'].notna()]
        ldtime.Demora_avg = ldtime.Demora_avg.round()
        ldtime.LDtime_avg = ldtime.LDtime_avg.round()
        dic_prom = ldtime.set_index('Material')['Demora_avg'].to_dict()
        dic_st = ldtime.set_index('Material')['LDtime_avg'].to_dict()

        abcd = for_abc[['skus', 'service_level']]
        for_reorder = pd.merge(groupedd, abcd, how='left', left_on='Material', right_on='skus')
        for_reorder.Material = for_reorder.Material.astype(str)
        for_reorder['leadtime_prom'] = for_reorder.Material.map(dic_prom)
        for_reorder['leadtime_sd'] = for_reorder.Material.map(dic_st)
        for_reorder['leadtime_prom'] = for_reorder['leadtime_prom'].fillna(for_reorder['leadtime_prom'].mean())
        for_reorder['leadtime_sd'] = for_reorder['leadtime_sd'].fillna(for_reorder['leadtime_sd'].mean())

        empty_data = pd.DataFrame()
        for i in range(for_reorder.shape[0]):
            ordering_point = inv.reorderpoint(for_reorder.loc[i, 'promedio'], for_reorder.loc[i, 'sd'], for_reorder.loc[i, 'leadtime_prom'], for_reorder.loc[i, 'service_level'])
            as_data = pd.DataFrame(ordering_point, index=[0])
            as_data['Material'] = for_reorder.loc[i, 'Material']
            empty_data = pd.concat([empty_data, as_data], axis=0)

        all_data = pd.merge(for_reorder, empty_data, how='left')
        all_data['saftey_stock'] = all_data['reorder_point'] - all_data['demandleadtime']
        all_data = all_data[all_data.saftey_stock != max(all_data.saftey_stock)]
        all_data = all_data.drop(columns=['skus', 'leadtime_sd', 'demandleadtime', 'sigmadl', 'sd', 'promedio'])
        
        # Añadir la denominación a all_data
        all_data = pd.merge(all_data, df[['Material', 'Denominacion']].drop_duplicates(), on='Material', how='left')
        
        # Añadir la denominación a for_abc
        for_abc = pd.merge(for_abc, df[['Material', 'Denominacion']].drop_duplicates(), left_on='skus', right_on='Material', how='left').drop(columns=['Material'])
        ultima_columna = for_abc.columns[-1]
        # Crear una lista con el nuevo orden de las columnas
        nuevo_orden = [for_abc.columns[0], ultima_columna] + list(for_abc.columns[1:-1])
        
        # Reordenar las columnas del DataFrame
        for_abc = for_abc[nuevo_orden]
        #####################################3
        ultima_columna = all_data.columns[-1]

        # Crear una lista con el nuevo orden de las columnas
        nuevo_orden = [all_data.columns[0], ultima_columna] + list(all_data.columns[1:-1])

        # Reordenar las columnas del DataFrame
        all_data = all_data[nuevo_orden] 
        all_data['reorder_point'] = all_data['reorder_point'].round()
        all_data['saftey_stock'] = all_data['saftey_stock'].round()  
        st.table(for_abc.head(5))
        
        

        # Función para convertir DataFrame a Excel y devolver el archivo en bytes
        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'})
                worksheet.set_column('A:A', None, format1)
            processed_data = output.getvalue()
            return processed_data

        df_all_data_xlsx = to_excel(all_data)
        
        
        st.download_button(
            label='📥 Descargar Stock de seguridad y Puntos de reorden',
            data=df_all_data_xlsx,
            file_name='df_all_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    else:
        st.write("Por favor, sube ambos archivos para continuar.")
        
        
        

# Función para la opción "DataFrame"
def mostrar_dataframe():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import datetime
    import inventorize as inv
    import io
    from io import BytesIO

    st.sidebar.header("Cargar archivo de ventas")
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de ventas", type=["csv"])
    # Lista de Familias
    um = [
        "M", "UN"
    ]

    # Menú desplegable para seleccionar la familia
    um1 = st.sidebar.selectbox("Selecciona la unidad de medida", um)
    if uploaded_file is not None:
        # Lectura de los archivos subidos
        dataset = pd.read_csv(uploaded_file, index_col=0, encoding='latin-1')
       
        # Procesamiento de datos
        df = dataset.drop(columns=[
            'Organización de ventas', 'Pagador', 'Nombre', 'Rut', 'Gr.Clt-CLI',
            'Gr.clientes-CLI', 'Gr.Ven.-CLI', 'Gr. Vendedores - CLI',
            'Gr.Precio-CLI', 'Nº pedido cliente', 'Fe.PedCli', 'Factura',
            'Referencia', 'Clase de factura', 'ZonaVen',
            'Zona de ventas', 'GI', 'Grupo de imputación', 'CPag', 'FeValFijad',
            'Referencia de pago', 'StatusC', 'An.', 'Pos.Fac',
            '% margen s/venta', '% margen s/cost', 'Subtotal 6', 'Mon..5', 'Gr.Cl1',
            'Gr.Clientes 1', 'Grupo art', 'Grupo de artículos', 'GrMat',
            'Grupo materiales', 'GImMat', 'Gr.Imp.Material', 'OfVta',
            'Oficina de ventas', 'Doc.venta', 'Tp.DC', 'Pos.Ventas', 'Gr.Ven.-PED',
            'Gr.vendedores-PED', 'Gr.Cl-PED', 'Gr.clientes-PED', 'GP-PED',
            'Gr.Precio-PED', 'Costes internos posición', 'Mon..6', 'Proc.empres.',
            'CeBe'
        ])
        df['Fecha factura'] = pd.to_datetime(df['Fecha factura'])
        df.rename(columns={'Fecha factura': 'fecha'}, inplace=True)
        df['Año'] = df.fecha.dt.year
        df = df[df['fecha'] >= '2022-01-1']
        df.rename(columns={'Denominación': 'Denominacion', 'Ctd.facturada2': 'Unidad'}, inplace=True)
        df = df[~df['Denominacion'].str.contains('POLIMERO|CLORURO|SULFATO|FLETE|UREA|REACTIVO')]
        df['Cantidad'] = df['Cantidad'].str.replace(",", "").str.replace(".", "")
        df['NETO'] = df['NETO'].str.replace(",", "").str.replace(".", "")
        df['Margen'] = df['Margen'].str.replace(",", "").str.replace(".", "")
        df = df[df['Cantidad'].notna()]
        df['Material'] = df['Material'].astype(str)
        df['Cantidad'] = df['Cantidad'].astype(int)
        df['NETO'] = df['NETO'].astype(int)
        df['Margen'] = df['Margen'].astype(int)
        df = df[~df['Material'].str.startswith('S', na=False)]
        df['Familia'] = df['Denominacion'].str.split().str[0]
        df['date'] = df.fecha.dt.strftime('%Y-%m-%d')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['UM'] == um1]
        abc=df
        st.sidebar.header("Filtro de Fecha para ABC")
        start_date = st.sidebar.date_input("Fecha de inicio", value=abc['date'].min())
        end_date = st.sidebar.date_input("Fecha de fin", value=abc['date'].max())
        
        # Aplicar filtro de fecha
        abc_filtered = abc[(abc['date'] >= pd.to_datetime(start_date)) & (abc['date'] <= pd.to_datetime(end_date))]
        ventas_group = abc_filtered.groupby('Denominacion').agg(codigo_material=('Material', np.unique), cantidad_total= ('Cantidad',np.sum),unidad_medida=('UM', np.unique),ventas_totales= ('NETO',np.sum), margen_total =('Margen', np.sum),familia=('Familia', np.unique)).reset_index()
        ventas_group.familia=ventas_group.familia.astype(str)
        ventas_group.unidad_medida=ventas_group.unidad_medida.astype(str)
        ventas_group.codigo_material=ventas_group.codigo_material.astype(str)
        b=inv.productmix(ventas_group['Denominacion'], ventas_group['ventas_totales'],ventas_group['margen_total'])
        b.head(20)
        df_group = df.groupby(['Denominacion', 'fecha']).agg(ventas_totales=('NETO', 'sum')).reset_index()

        cv_data = df_group.groupby('Denominacion').agg(promedio_ventas=('ventas_totales', 'mean'), sd_ventas=('ventas_totales', 'std')).reset_index()
        cv_data['cv_squared'] = (cv_data['sd_ventas'] / cv_data['promedio_ventas']) ** 2

        producto_date = df_group.groupby('Denominacion').agg(cantidad=('Denominacion', 'count')).reset_index()
        skus = producto_date.Denominacion.unique()
        fecha = max(df.date)
        nuevafecha = fecha - datetime.timedelta(days=30 * 4)
        last_four = df[df.fecha > nuevafecha]
        last_four['month'] = last_four.date.dt.month
        last_four['year'] = last_four.date.dt.year
        last_four['week'] = last_four.date.dt.isocalendar().week

        aa = last_four.groupby(['week', 'month', 'year', 'Material']).agg(
            nom_material=('Denominacion', np.unique), date=('date', 'first'), total_semanal=('Cantidad', np.sum), total_ventas=('NETO', np.sum)).reset_index()

        groupedd = aa.groupby('Material').agg(promedio=('total_semanal', np.mean), sd=('total_semanal', 'std'), total_demanda=('total_semanal', np.sum), total_ventas=('total_ventas', np.sum)).reset_index()
        for_abc = inv.productmix(groupedd['Material'], groupedd['total_demanda'], groupedd['total_ventas'])

        mapping = {'A_A': 0.95, "A_C": 0.95, "C_A": 0.8, "A_B": 0.95, 'B_A': 0.7, "B_C": 0.75, "C_C": 0.7, "B_B": 0.7, "C_B": 0.8}
        for_abc['service_level'] = for_abc.product_mix.map(mapping)

        abcd = for_abc[['skus', 'service_level']]
        

        # Añadir la denominación a for_abc
        for_abc = pd.merge(for_abc, df[['Material', 'Denominacion']].drop_duplicates(), left_on='skus', right_on='Material', how='left').drop(columns=['Material'])
        ultima_columna = for_abc.columns[-1]
        # Crear una lista con el nuevo orden de las columnas
        nuevo_orden = [for_abc.columns[0], ultima_columna] + list(for_abc.columns[1:-1])
        
        # Reordenar las columnas del DataFrame
        for_abc = for_abc[nuevo_orden]

        # Reordenar las columnas del DataFrame  
        st.markdown("ABC VENTAS")
        st.table(b.head(5))
        st.markdown("ABC UNIDADES")
        st.table(for_abc.head(5))
        
        

        # Función para convertir DataFrame a Excel y devolver el archivo en bytes
        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'})
                worksheet.set_column('A:A', None, format1)
            processed_data = output.getvalue()
            return processed_data

        df_for_abc_xlsx = to_excel(for_abc)
        df_b_xlsx = to_excel(b)
        
        
        st.download_button(
            label='📥 Descargar ABC de unidades vendidas',
            data=df_for_abc_xlsx,
            file_name='df_for_abc.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        st.download_button(
            label='📥 Descargar ABC de Margen de ventas',
            data=df_b_xlsx,
            file_name='df_b.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.write("Por favor, sube ambos archivos para continuar.")
    def Ranking():
        st.sidebar.header("Cargar archivo de ventas")
        uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de ventas", type=["csv"])

# Mostrar contenido basado en la opción seleccionada
if option == 'Forecast':
    hola_mundo()
elif option == 'Stock de Seguridad y Punto de Reorden':
    mostrar_grafico()
elif option == 'ABC':
    mostrar_dataframe()
elif option == 'Ranking':
    Ranking()
