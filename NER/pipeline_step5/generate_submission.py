import json
import csv

def read_json(filename):
    a = []
    with open(filename) as f:
        for line in f:
            a.append(json.loads(line))

    return a

def read_result(filename):
    a = []
    with open(filename,'r') as f:
        for line in f:
            a.append(line.split())
    return a

def output_csv(filename, header, data):
    with open(filename, 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerow(header)
        csv_writer.writerows(data)

def get_index(tag):
    temp = []
    temp_tag = []
    IDs = []
    Tags = []
    entity = ['DiseaseOrPhenotypicFeature', 'ChemicalEntity', 'OrganismTaxon', 'GeneOrGeneProduct', 'SequenceVariant', 'CellLine']
    flag = [0 for i in range (len(entity))]
    for i, x in enumerate(tag):
        if x == 'O':
            if temp != []:
                IDs.append(temp)
                Tags.append(temp_tag)
            temp = []
            temp_tag = []
            flag = [0 for i in range (len(entity))]
        else:
            if flag[entity.index(x.split('-')[1])] == 1:
                temp.append(i)
                temp_tag.append(x.split('-')[1])
            else:
                if temp != []:
                    IDs.append(temp)
                    Tags.append(temp_tag)
                temp = [i]
                temp_tag = [x.split('-')[1]]
                flag = [0 for i in range (len(entity))]
                flag[entity.index(x.split('-')[1])] = 1
        if i==len(tag)-1 and temp != []:
            IDs.append(temp)
            Tags.append(temp_tag)
    return IDs, Tags
            
def refine_tag_fixGap(input_tag, entity, gap=1):
    '''
    refine: 'OIB' to 'OBI', 'OBB' to 'OBI', 'OBIB' to 'OBII', 'BOI' to 'BII', 'BIOI' to 'BIII'. Here B and I are same entity type.
     input: input_tag is a list of tags within 6 entities, e.g. ['I-ChemicalEntity', 'O', 'I-ChemicalEntity','I-ChemicalEntity','O']
    output: refined_tag is a list of refined tags, e.g. ['B-ChemicalEntity', 'O', 'B-ChemicalEntity','I-ChemicalEntity','O']
    '''
    gap = int(gap)
    full_refined_tag = []
    for sent in input_tag:
        refined_tag = []
        #entity = ['DiseaseOrPhenotypicFeature', 'ChemicalEntity', 'OrganismTaxon', 'GeneOrGeneProduct', 'SequenceVariant', 'CellLine']
        flag = [0 for i in range (len(entity))]        # flag use to decide if asign B or I
        for candidate in sent:
            if candidate == 'O':    # the top candidate is "O"
                refined_tag.append('O')
                # the flag of any non-zero entity will add 1
                flag =  [x+1 if x!=0 else 0 for x in flag]
            elif candidate.split('-')[0] == 'B':
                if flag[entity.index(candidate.split('-')[1])] != 1: # the previous position "O", assign "B"
                    refined_tag.append('B-'+candidate.split('-')[1])
                    flag = [0 for j in range (len(entity))] # reset flag
                    flag[entity.index(candidate.split('-')[1])] = 1
                else: # the previous position "B" or "I", assign "I"
                    refined_tag.append('I-'+candidate.split('-')[1])
                    flag = [0 for j in range (len(entity))] # reset flag
                    flag[entity.index(candidate.split('-')[1])] = 1
            elif candidate.split('-')[0] == 'I':
                if flag[entity.index(candidate.split('-')[1])] == 1:    # the previous position 'I' or 'B'
                    refined_tag.append('I-'+candidate.split('-')[1])
                elif flag[entity.index(candidate.split('-')[1])] >gap+1:    # the gap larger than gap_cutoff, assign 'B'
                    refined_tag.append('B-'+candidate.split('-')[1])
                    flag = [0 for j in range (len(entity))] # reset flag
                    flag[entity.index(candidate.split('-')[1])] = 1
                elif flag[entity.index(candidate.split('-')[1])] == 0:      # this type happends first time
                    refined_tag.append('B-'+candidate.split('-')[1])
                    flag = [0 for j in range (len(entity))] # reset flag
                    flag[entity.index(candidate.split('-')[1])] = 1
                else:
                    for k in range (1, flag[entity.index(candidate.split('-')[1])]):    # fill the gap
                        refined_tag[-k] = 'I-'+candidate.split('-')[1]
                    refined_tag.append('I-'+candidate.split('-')[1])
                    flag = [0 for j in range (len(entity))] # reset flag
                    flag[entity.index(candidate.split('-')[1])] = 1
        full_refined_tag.append(refined_tag)
    return full_refined_tag

def output_refine_tag(result, filename):
    with open(filename,'w') as f:
        f.writelines('\n'.join([' '.join(x) for x in result])+'\n')

if __name__ == "__main__":
    import sys
    input_file = "Litcoin_testset.json"
    predict_file = sys.argv[1]
    entity = ['DiseaseOrPhenotypicFeature', 'ChemicalEntity', 'OrganismTaxon', 'GeneOrGeneProduct', 'SequenceVariant', 'CellLine']
    header = ['id','abstract_id', 'offset_start', 'offset_finish', 'type']
    info = read_json(input_file)
    result = read_result(predict_file)
    output_name = predict_file[:-4]
    output_refine_tag(result, output_name+'.txt')
    output = []
    abstract_flag = []
    entity_id = -1
    for i, x in enumerate(info):
        abstract_id = x['document_id'][0]
        span = x['spans']
        tag = result[i]
        IDs, Tags = get_index(tag)
        if abstract_id not in abstract_flag:
            entity_id = -1
            abstract_flag.append(abstract_id)
        for j in range (len(IDs)):
            entity_id += 1
            output.append([entity_id, abstract_id, span[IDs[j][0]][0], span[IDs[j][-1]][-1], list(set(Tags[j]))[0]])
    output_csv(output_name+'.csv', header, output)


