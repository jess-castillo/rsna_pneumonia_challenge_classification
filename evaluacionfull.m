function [matriz_confusion_normalizada,ACA] = evaluacionfull(model,hist_imp,anotaciones_test)
%Demo de la clasificación de la imagen completa. La función recibe una
%imagen, el histograma a imponer, las anotaciones y el modelo de
%clasificación y devuelve una etiqueta.
global dimf nbins;
% Nombres de las imágenes de prueba:
names_test = extractfield(dir(fullfile('Database','test','*.dcm')),'name');

histogramas_test = zeros(length(names_test),nbins);
anotacion_histo_test = zeros(length(names_test),1);
for i=1:length(names_test)
    %Se lee la imagen:
    im0 = dicomread(fullfile('Database','test',names_test{i}));
    %Se redimensiona la imagen:
    im0 = imresize(im0,[dimf dimf]);
    %Se aplica le apliza la ecualización adaptativa:
    im = adapthisteq((im0));
    %Se impone el histograma descargado:
    im_imposition = histeq(im,hist_imp);
    
    %% Umbralización de Otsu:
    % Umbralización de la imagen ecualizada adaptativa + Imposición:
    level_im = multithresh(adapthisteq(im_imposition),2);
    seg_I_im = imquantize(adapthisteq(im_imposition),level_im);
    %im_eqim_otsu = label2rgb(seg_I_im);
    
    %% Extracción de la máscara del pulmón:
    %Se crea la máscara con la imagen impuesta(adaptativa) umbralizada.
    mask = seg_I_im == 1;
    
    %Se crea el elemento estructurante con el cual se realizará la dilatación
    %geodésica:
    se = strel('disk',10);
    
    % Eliminación de objetos que tocan el borde por dilatación geodésica.
    %Se crea el marcador:
    marcador = imerode(mask,se);
    % Se reconstruye la imagen:
    im_reconstruct = imreconstruct(marcador,mask);
    %Se eliminan los bordes:
    sin_borde = imclearborder(im_reconstruct);
    
    %% Operaciones morfológicas para sacar la máscara:
    %Se hace apertura para eliminar algunos cosas que no son pulmón:
    ap = imopen(sin_borde,strel('disk',20));
    %Se dilata con otro elemento estructurante para probrar relleno:
    dil = imdilate(ap,se);
    %Reconstucción geodésica para probar relleno:
    mask_final = imreconstruct(dil,sin_borde);
    
    %Se extrae la región del pulmon usando la máscara:
    extraccion = uint8(mask_final).*im0;
    %Se obtiene un histograma de la imagen extraída:
    aux = imhist(extraccion,nbins)/sum(size(extraccion,1)*size(extraccion,2));
    histogramas_test(i,:) = aux(:)';
    anotacion_histo_test(i) = anotaciones_test{i,6};
end
%Se predice la etiqueta de la imagen:
test_pred = predict(model,histogramas_test);
test_pred = str2num(cell2mat(test_pred));

%Se calcula la matriz de confusión:
matriz_confusion = confusionmat(anotacion_histo_test,test_pred)';
matriz_confusion_normalizada = matriz_confusion./sum(matriz_confusion,1);
%Se calcula el ACA:
ACA = (sum(diag(matriz_confusion_normalizada))/2)*100;
end
