function [matriz_confusion_normalizada,ACA] = evaluacionbbox(model,anotaciones_test)
%Evaluación de la clasificación sobre los bounding boxes.
%La función recibe por parámetro el modelo y las anotaciones, y devuelve
%la matriz de confusión y el ACA del método.
global dimb xn yn wn hn nbins;
% Nombres de las imágenes de prueba:
names_test = extractfield(dir(fullfile('Database','test','*.dcm')),'name');

ann_test = zeros(1,length(names_test));
histogramas = zeros(length(names_test),nbins);
for i= 1: length(anotaciones_test)
    % Si la imagen de train que se está evaluando está anotada como
    % positiva para neumonia
    if anotaciones_test{i,6} == 1;
        % Se obtienen las coordenadas del bounding box de dicha imagen
        x = anotaciones_test{i,2};
        y = anotaciones_test{i,3};
        w = anotaciones_test{i,4};
        h = anotaciones_test{i,5};
        % Se guardan dichas coordenadas en un vector
        rec = [ x y w h];
        
        im_actual = dicomread(fullfile('Database','test',names_test{i}));
        
        % Se saca el bounding box de dicha imagen.
        bounding = imcrop(im_actual,rec);
        % Se redimensiona el bounding box de dicha imagen.
        nueva_imagen = imresize(bounding,[dimb dimb]);
        % Se llena una nueva celda donde se guarda la información de la
        % nueva imagen
        aux = imhist(nueva_imagen,nbins)/sum(size(nueva_imagen,1)*size(nueva_imagen,2));
        histogramas(i,:) = aux(:)';
        ann_test(i) = anotaciones_test{i,6};
    else anotaciones_test{i,6} == 0;
        rec = [xn yn wn hn];
        % Se obtiene el nombre de la imagen i-esima de la carpeta de train
        im_actual = dicomread(fullfile('Database','test',names_test{i}));
        % Se saca el bounding box de dicha imagen.
        bounding = imcrop(im_actual,rec);
        % Se redimensiona el bounding box de dicha imagen.
        nueva_imagen = imresize(bounding,[dimb dimb]);
        % Se llena una nueva celda donde se guarda la información de la
        % nueva imagen
        aux = imhist(nueva_imagen,nbins)/sum(size(nueva_imagen,1)*size(nueva_imagen,2));
        histogramas(i,:) = aux(:)';
        ann_test(i) = anotaciones_test{i,6};
    end
end

%Se predicen las etiquetas de test:
test_pred = predict(model,histogramas);
test_pred = str2num(cell2mat(test_pred));

%Se calcula la matriz de confusión:
matriz_confusion = confusionmat(ann_test,test_pred)';
matriz_confusion_normalizada = matriz_confusion./sum(matriz_confusion,1);
%Se calcula el ACA:
ACA = (sum(diag(matriz_confusion_normalizada))/2)*100;

end

