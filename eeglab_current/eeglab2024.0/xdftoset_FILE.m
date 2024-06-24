function xdftoset_FILE(file)
chdir(file);
LIST_xdf = ls('*.xdf');
LIST_set= ls('*.set');
OUTPUT_xdf=extractFilenamesCharArray(LIST_xdf);
OUTPUT_set=extractFilenamesCharArray(LIST_set);
for i = 1:length(OUTPUT_xdf)
    output = OUTPUT_xdf(i);
    if not(any(contains(OUTPUT_set,output)))
        xdftoset(file,char(output));
    end
end
