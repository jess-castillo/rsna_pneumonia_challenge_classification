function [label,nueva_imagen] = demobbox(im,model,anotacion)
%Demo de la clasificación de los bounding boxes. La función recibe una
%imagen, las anotaciones y el modelo de clasificación y devuelve una 
%etiqueta y la región de la imagen evaluada.
global dimb xn yn wn hn nbins;
%Se evalúa la anotación de test con el fin de obtener el bounding box:
 if anotacion{1,6} == 1;
        % Se obtienen las coordenadas del bounding box de dicha imagen
        x = anotacion{1,2};
        y = anotacion{1,3};
        w = anotacion{1,4};
        h = anotacion{1,5};
        % Se guardan dichas coordenadas en un vector
        rec = [ x y w h];
        
        % Se saca el bounding box de la imagen.
        bounding = imcrop(im,rec);
        % Se redimensiona el bounding box de dicha imagen.
        nueva_imagen = imresize(bounding,[dimb dimb]);
        % Se almacena la información del histograma del bounding box
        aux = imhist(nueva_imagen,nbins)/sum(size(nueva_imagen,1)*size(nueva_imagen,2));
        histograma = aux(:)';
    else anotacion{1,6} == 0;
        %Se define el bounding box:
        rec = [xn yn wn hn];
        %Se saca el bounding box de la imagen.
        bounding = imcrop(im,rec);
        % Se redimensiona el bounding box de dicha imagen.
        nueva_imagen = imresize(bounding,[dimb dimb]);
        % Se almacena la información del histograma del bounding box
        aux = imhist(nueva_imagen,nbins)/sum(size(nueva_imagen,1)*size(nueva_imagen,2));
        histograma = aux(:)';
 end
label = predict(model,histograma);
label = str2double(cell2mat(label));
end 

