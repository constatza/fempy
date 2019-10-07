


class GenericDOFEnumerator:
    """Retrieves element connectivity data required for matrix assembly."""
    
    @staticmethod
    def get_DOF_types(element):
        """Retrieves the dof types of each node."""
        return element.element_type.get_element_DOFtypes(element)
    
    @staticmethod
    def get_DOFtypes_for_DOF_enumeration(element):
        """Retrieves the dof types of each node."""
        return element.element_type.get_element_DOFtypes(element)

    @staticmethod
    def get_nodes_for_matrix_assembly(element):
        """Retrieves the element nodes."""
        return element.nodes

    @staticmethod
    def get_transformed_matrix(matrix):
        """Retrieves matrix transformed from local to global coordinate system."""
        return matrix

    @staticmethod
    def get_tranformed_displacements_vector(vector):
        """ Retrieves displacements transformed from local to global coordinate system."""
        return vector
    
    def get_transformed_forces_vector(vector):
        """Retrieves displacements transformed from local to global coordinate system."""
        return vector


class GlobalMatrixAssembler:
    """Assembles the global  matrix."""
    
    @staticmethod           
    def calculate_global_matrix(model, element_provider, nodal_DOFs_dictionary=None):
        """Calculates the generic global matrix. The type of matrix i.e. stiffness,
        mass etc. is defined by the type of the element_provider."
       
        Parameters
        ----------
        model : Model
            The model whose matrix is to be calculated.
        nodal_DOFs_dictionary : dict<int, dict<DOFType, int>>
            Dictionary that links node.ID and DOFType with the equivalent 
            global nodal DOF number.
        element_provider : ElementStiffnessProvider
            The element provider.
        
        Returns
        -------
        global_stiffness_matrix : np.ndarray
            Model global stiffness matrix.
        """
        
        if nodal_DOFs_dictionary is None:
            nodal_DOFs_dictionary = model.nodal_DOFs_dictionary
        
        numels = model.number_of_elements
        globalDOFs = empty((numels, 8), dtype=int)
        total_element_matrices = empty((8, 8, numels))
                
        #!!!!! Change if using different type of elements
        element = model.elements[0]
        get_DOF_types = element.element_type.DOF_enumerator.get_DOF_types
        get_nodes_for_matrix_assembly = element.element_type.DOF_enumerator.get_nodes_for_matrix_assembly
        for k,element in enumerate(model.elements):
            
            total_element_matrices[:, :, k] = element_provider.matrix(element)
            is_first_analysis = model.global_DOFs is None
            if is_first_analysis: 
                element_DOFtypes = get_DOF_types(element)
                matrix_assembly_nodes = get_nodes_for_matrix_assembly(element)
                
                counter = -1
                for i in range(len(element_DOFtypes)):
                    node = matrix_assembly_nodes[i]
                    for DOFtype in element_DOFtypes[i]:
                        counter += 1
                        globalDOFs[k, counter] = nodal_DOFs_dictionary[node.ID][DOFtype]
                        
            else:
                globalDOFs = model.global_DOFs

        numDOFs = model.total_DOFs
        globalDOFs = globalDOFs.astype(int)
        global_matrix = GlobalMatrixAssembler.assign_element_to_global_matrix(
                                                                        total_element_matrices,
                                                                        globalDOFs,
                                                                        numDOFs)                                                           
        model.global_DOFs = globalDOFs
        return global_matrix 
                
    @staticmethod
    @njit('float64[:, :](float64[:, :, :], int32[:, :], int64)')
    def assign_element_to_global_matrix(element_matrices, globalDOFs, numDOFs):
        K = zeros((numDOFs, numDOFs))
        for ielement in range(element_matrices.shape[2]):
            for i in range(8):
                DOFrow = globalDOFs[ielement, i]
                if DOFrow != -1:
                    for j in range(i, 8):
                        DOFcol = globalDOFs[ielement, j]
                        if DOFcol != -1:
                            K[DOFrow, DOFcol] += element_matrices[i, j, ielement]
                            K[DOFcol, DOFrow] = K[DOFrow, DOFcol]                 
        return K
