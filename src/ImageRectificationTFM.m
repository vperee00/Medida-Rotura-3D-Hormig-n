clear all;
close all;

warning('off', 'all');

disp('Inicio...');

indice=685;
par_de_imagenes="0" + indice +"";

for indice = 1:685
    par_de_imagenes = sprintf("%04d", indice);
    
    disp('Calibrando par ' + par_de_imagenes);

    % Especifica la ruta del directorio
    directorio = "Y:\" + par_de_imagenes + "\";

    % Verifica si el directorio existe
    if exist(directorio, 'dir') ~= 7
        disp('Creando directorio ' + directorio);
        mkdir(directorio);
    end

    load CalibrationParametersSession17Dec2024

    I1End=imread("Z:\Imagenes\Experimento\capturas_cam1\cam1_frame" + par_de_imagenes + ".png");
    I2End=imread("Z:\Imagenes\Experimento\capturas_cam2\cam2_frame" + par_de_imagenes + ".png");

    [I1EndRect, I2EndRect] = rectifyStereoImages(I1End, I2End, stereoParams17Dec2024);

    % Rota las imágenes 180 grados
    I1EndRectRotated = imrotate(I1EndRect, 180);
    I2EndRectRotated = imrotate(I2EndRect, 180);

    imwrite(uint8(I1EndRectRotated),"Y:\" + par_de_imagenes + "\left_" + par_de_imagenes + ".png");
    imwrite(uint8(I2EndRectRotated),"Y:\" + par_de_imagenes + "\right_" + par_de_imagenes + ".png");

end


disp('Final');