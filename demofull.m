function [label] = demofull(im,model,hist_imp)
%Demo de la clasificación de la imagen completa. La función recibe una
%imagen, el histograma a imponer, las anotaciones y el modelo de
%clasificación y devuelve una etiqueta.
global dimf nbins;
%Se redimensiona la imagen:
im0 = imresize(im,[dimf dimf]);
%Se aplica le apliza la ecualización adaptativa:
im = adapthisteq((im0));
%Se impone el histograma descargado:
im_imposition = histeq(im,hist_imp);

%% Umbralización de Otsu:
% Umbralización de la imagen ecualizada adaptativa + Imposición:
level_im = multithresh(adapthisteq(im_imposition),2);
seg_I_im = imquantize(adapthisteq(im_imposition),level_im);
im_eqim_otsu = label2rgb(seg_I_im);

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
histograma = aux(:)';

%Se predice la etiqueta de la imagen:
label = predict(model,histograma);
label = str2double(cell2mat(label));
end

