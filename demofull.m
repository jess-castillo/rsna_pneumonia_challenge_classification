function [label] = demofull(im,model,hist_imp)
%Demo de la clasificaci�n de la imagen completa. La funci�n recibe una
%imagen, el histograma a imponer, las anotaciones y el modelo de
%clasificaci�n y devuelve una etiqueta.
global dimf nbins;
%Se redimensiona la imagen:
im0 = imresize(im,[dimf dimf]);
%Se aplica le apliza la ecualizaci�n adaptativa:
im = adapthisteq((im0));
%Se impone el histograma descargado:
im_imposition = histeq(im,hist_imp);

%% Umbralizaci�n de Otsu:
% Umbralizaci�n de la imagen ecualizada adaptativa + Imposici�n:
level_im = multithresh(adapthisteq(im_imposition),2);
seg_I_im = imquantize(adapthisteq(im_imposition),level_im);
im_eqim_otsu = label2rgb(seg_I_im);

%% Extracci�n de la m�scara del pulm�n:
%Se crea la m�scara con la imagen impuesta(adaptativa) umbralizada.
mask = seg_I_im == 1;

%Se crea el elemento estructurante con el cual se realizar� la dilataci�n
%geod�sica:
se = strel('disk',10);

% Eliminaci�n de objetos que tocan el borde por dilataci�n geod�sica. 
%Se crea el marcador:
marcador = imerode(mask,se);
% Se reconstruye la imagen:
im_reconstruct = imreconstruct(marcador,mask);
%Se eliminan los bordes:
sin_borde = imclearborder(im_reconstruct);

%% Operaciones morfol�gicas para sacar la m�scara:
%Se hace apertura para eliminar algunos cosas que no son pulm�n:
ap = imopen(sin_borde,strel('disk',20));
%Se dilata con otro elemento estructurante para probrar relleno:
dil = imdilate(ap,se);
%Reconstucci�n geod�sica para probar relleno:
mask_final = imreconstruct(dil,sin_borde);

%Se extrae la regi�n del pulmon usando la m�scara:
extraccion = uint8(mask_final).*im0;
%Se obtiene un histograma de la imagen extra�da:
aux = imhist(extraccion,nbins)/sum(size(extraccion,1)*size(extraccion,2));
histograma = aux(:)';

%Se predice la etiqueta de la imagen:
label = predict(model,histograma);
label = str2double(cell2mat(label));
end

