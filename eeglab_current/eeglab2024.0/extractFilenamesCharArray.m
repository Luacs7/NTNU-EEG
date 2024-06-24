function filenames = extractFilenamesCharArray(charArray)
    % Cette fonction prend un tableau de caractères n*N de noms de fichiers
    % et retourne une cellule de noms de fichiers sans l'extension .set

    % Nombre de fichiers
    numFiles = size(charArray, 1);
    
    % Initialiser la cellule de noms de fichiers sans extension
    filenames = cell(numFiles, 1);
    
    for i = 1:numFiles
        % Obtenir le nom du fichier actuel en supprimant les espaces de fin
        file = strtrim(charArray(i, :));
        
        % Trouver l'indice du dernier point dans le nom du fichier
        dotIndex = strfind(file, '.');
        
        % Si un point est trouvé
        if ~isempty(dotIndex)
            % Extraire la partie avant le dernier point
            filenames{i} = file(1:dotIndex(end)-1);
        else
            % Si pas de point trouvé, retourner le nom du fichier tel quel
            filenames{i} = file;
        end
    end
end
