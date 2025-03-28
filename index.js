document.getElementById("predictButton").addEventListener("click", async function() {
    const texto = document.getElementById("texto").value;
    const pais = document.getElementById("pais").value;
    const tema = document.getElementById("tema").value;
    const fuentes = parseInt(document.getElementById("numero_fuentes").value, 10);
    const estadisticas = parseInt(document.getElementById("numero_estadisticas").value, 10);
    const adjetivos = parseInt(document.getElementById("numero_adjetivos").value, 10);
    const t_ideologicos = parseInt(document.getElementById("numero_terminos").value, 10);
    const palabras = parseInt(document.getElementById("numero_palabras").value, 10);
    const imagenes = parseInt(document.getElementById("numero_imagenes").value, 10);
    const citas = parseInt(document.getElementById("numero_citas_directas").value, 10);
    const reconocido = parseInt(document.querySelector("input[name='reconocido']:checked").value, 10)
    const especializado = parseInt(document.querySelector("input[name='especializado']:checked").value, 10)
    const tono = parseInt(document.querySelector("input[name='tono']:checked").value, 10)

    console.log(tema, fuentes, estadisticas, adjetivos, t_ideologicos, palabras, imagenes, citas, reconocido, especializado, tono)
    
    const payload = {
        texto: texto,
        datos_estructurados: {
            numero: 1,
            fuentes: fuentes,
            estadisticas: estadisticas,
            numero_adjetivos: adjetivos,
            terminos_ideologicos: t_ideologicos,
            numero_palabras: palabras,
            imagenes: imagenes,
            citas_directas: citas,
            medio_reconocido: reconocido,
            medio_especializado: especializado,
            formalidad: tono,
            emocionalidad: tono == 1 ? 0 : 1,
            pais_de_origen: pais,
            tema_principal: tema
        }
    };

    console.log(payload)

    document.getElementById("predictButton").disabled = true;
    
    try {
        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        document.getElementById("resultado").innerHTML = "<strong>Predicción:</strong> " + data.prediccion_principal.categoria;
        document.getElementById("resultado").style.display = "block";
        console.log(data)
    } catch (error) {
        alert("Error al obtener la predicción");
    } finally {
        document.getElementById("predictButton").disabled = false;
    }
});