%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
                An�lisis y Procesamiento de Im�genes: Proyecto Final.
            %Clasificaci�n binaria de neumon�a a partir de rayos X de t�rax
                                    %Autores:
                              Alejandra Barrera Suarez.
                            ma.barrera20@uniandes.edu.co

                              Jessica Castillo Rojas.
                            jl.castillo@uniandes.edu.co

                                Juan David Garc�a.
                            jd.garcia20@uniandes.edu.co
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Se descargan las bases de datos y archivos archivos necesarios:

% Se descarga una radiograf�a con alto contraste para imponerla a las
%im�genes:
url = 'https://drive.google.com/uc?authuser=0&id=1tX7owgI0hR45gmZTNTY9C_M-ycRmsfT_&export=download';
options = weboptions;
options.Timeout = 60;
filename = 'cxr.jpg';
outfilename = websave(filename,url);
%Se extrae el histograma a imponer:
hist_imp = imhist(rgb2gray(imread('cxr.jpg')));
disp('Imagen de imposici�n descargada.');


%Se descarga la base de datos de prueba con el archivo de anotaciones:
url2 = 'https://www.dropbox.com/s/w71c1jcx11sho84/Database.zip?dl=1';
options = weboptions;
options.Timeout = 60;
filename2 = 'Database_prueba.zip';
outfilename = websave(filename2,url2);
disp('Database de prueba descargada.');

unzip('Database_prueba.zip');
disp('Carpeta descomprimida.');
% Se descarga el modelo entrenado:
load data.mat;
%% 1. PREPARACI�N DE ARCHIVOS
% 1.1. Lectura de archivos:
% Nombres de las im�genes de entrenamiento, validaci�n y prueba:
names_test = extractfield(dir(fullfile('Database','test','*.dcm')),'name');

% 1.2 Obtener todas las anotaciones de test y separarlas:
[anotaciones] = readtable(fullfile('Database','Annotations.txt'),'HeaderLines',1);
%Se convierte la tabla a celdas:
anotaciones = table2cell(anotaciones);

%Se a�ade la extensi�n:
for i = 1:length(anotaciones)
    anotaciones{i,1} = strcat(anotaciones{i,1},'.dcm');
    anotaciones{i,1} = strip(anotaciones{i,1},'left','0');
end

% Anotaciones de test:
anotaciones_test = cell(length(names_test),6);
% Anotaciones de los nombres:
nombres_ann = anotaciones(:,1);
for i = 1:length(names_test)
    %Test:
    nombre_test_ac = names_test{i};
    nombre_test_ac = strip(nombre_test_ac,'left','0');
    
    idx3 = find(contains(nombres_ann,nombre_test_ac));
    anotaciones_test(i,:) = anotaciones(idx3,:);
end

%Se definen las coordenadas de bounding boxes para los negativos y la
%dimensi�n para los bounding boxes:
global dimb xn yn wn hn nbins dimf;
dimb = 500;
dimf = 800;
xn = 300;
yn = 300;
wn = 170;
hn = 450;
nbins = 16;

%% 2. CLASIFICACI�N SOBRE LOS BOUNDING BOXES:
% 2.1 Demo sobre bounding boxes:
% 2.1.1 Se carga una imagen aleatoria de test. Para ello:
%Se toma un n�mero aleatorio entre 1 y la cantidad de imagenes de test:
a = num2str(ceil(rand*((length(names_test)-1)+1)/1));
%Se extrae el nombre de la imagen del �ndice aleatorio:
nom = names_test{a};
%Se eliminan los ceros del nombre:
sin_cero =  strip(nom,'left','0');
idx = find(contains(anotaciones_test(:,1),sin_cero));
%Se extrae la anotaci�n:
anotacion = anotaciones_test(idx,:);
%Se lee la imagen de este nombre:
im = dicomread(fullfile('Database','test',nom));
% 2.1.2 Se llama el demo de la clasificaci�n sobre bounding boxes:
[label,nueva_imagen] = demobbox(im,model,anotacion);
% 2.1.3 Se grafica la imagen:
% 2.1.3.1 Se construye el t�tulo de la figura:
if anotacion{1,6} == 1;
    ann = 'Presencia de neumon�a.';
elseif anotacion{1,6} ==0;
    ann = 'Ausencia de neumon�a.';
end
if label == 1;
    pre = 'Presencia de neumon�a.';
elseif label ==0;
    pre = 'Ausencia de neumon�a.';
end
figure('Name','Clasificaci�n del bounding box de una imagen aleatoria')
imshow(nueva_imagen);
str = sprintf('La anotaci�n del bounding box es: %s\n La predicci�n del bounding box es: %s',ann,pre);
title(str,'FontSize',10);
pause;
%% 2. CLASIFICACI�N SOBRE LOS BOUNDING BOXES:
% 2.2. Evaluaci�n sobre bounding boxes:
% 2.2.1 Evaluaci�n sobre todo el conjunto de test:
[matriz_confusion_normalizada,ACA] = evaluacionbbox(model,anotaciones_test);

% 2.2.2 Se muestra la matriz de confusion normalizada:
f = figure('Name','Matriz de confusi�n normalizada de la clasificaci�n de bounding boxes');
uit = uitable(f,'Data',matriz_confusion_normalizada);
pause;
% 2.2.3 Se muestra el ACA obtenido:
aca=num2str(ACA);
f = msgbox(cat(2,{'El ACA del m�todo de clasificaci�n de bounding boxes es: '},strcat(aca,'%')),'Resultado');
ah = get( f, 'CurrentAxes' );
ch = get( ah, 'Children' );
set(f, 'position', [100 440 270 80]);
set( ch, 'FontSize', 10 );
pause;
%% 3. CLASIFICACI�N SOBRE LAS IM�GENES COMPLETAS:
%3.1 Demo de la clasificaci�n de la imagen completa
%3.1.1 Se llama el demo para evaluar la misma imagen aleatoria de antes, pero
%esta vez se toma completa:
[label2] = demobbox(im,full_model,anotacion);
% 3.1.2 Se grafica la imagen:
% 3.1.2.1 Se construye el t�tulo de la figura:
if label2 == 1;
    pre2 = 'Presencia de neumon�a.';
elseif label2 ==0;
    pre2 = 'Ausencia de neumon�a.';
end
figure('Name','Clasificaci�n de una imagen aleatoria completa')
imshow(im);
str = sprintf('La anotaci�n de la imagen es: %s\n La predicci�n de la imagen es: %s',ann,pre);
title(str,'FontSize',10);
pause;

%% 3. CLASIFICACI�N SOBRE LAS IM�GENES COMPLETAS:
%3.2 Evaluaci�n de la clasificaci�n de la imagen completa
% 3.2.1 Evaluaci�n sobre todo el conjunto de test:
[matriz_confusion_normalizada2,ACA2] = evaluacionfull(full_model,hist_imp,anotaciones_test);

% 2.2.2 Se muestra la matriz de confusion normalizada:
f = figure('Name','Matriz de confusi�n normalizada de la clasificaci�n de las im�genes completas');
uit = uitable(f,'Data',matriz_confusion_normalizada2);
pause;
% 2.2.3 Se muestra el ACA obtenido:
aca2=num2str(ACA2);
f = msgbox(cat(2,{'El ACA del m�todo de clasificaci�n de las im�genes completas:: '},strcat(aca2,'%')),'Resultado');
ah = get( f, 'CurrentAxes' );
ch = get( ah, 'Children' );
set(f, 'position', [100 440 270 80]);
set( ch, 'FontSize', 10 );
pause;
