using Combinatorics
using Eirene
using ProgressBars


# """ TODO: uncommenting leads to error
#   Construct_MCF(partitions, max_dim)

# Computes the Multiscale Clustering Filtration (MCF)[^1] for a sequence of partitions 
# `partitions` up to dimension `max_dim` given by ``K^m =  K^m=\{\sigma\subseteq X\; |\; \exists m^\prime \le m,\; 
# \exists C \in \mathcal{P}(m^\prime): \sigma\subseteq C\}``

# TODO: finish

# [^1]: Preprint incoming
# """
function Construct_MCF(partitions=Vector{Vector{Int64}}(), filtration_indices=Vector{Float64}(), max_dim=3)
    """Function constructs boundary matrix for MCF."""

    println("Construct MCF ... ")

    ##############
    # initialise #
    ##############

    # compute number of partitions
    n_partitions = size(partitions)[1]

    # initialise filtration indices if not given
    if size(filtration_indices)[1] < n_partitions
        filtration_indices = [1:n_partitions;]
    end

    # compute number of points and initialise keys
    n_points = size(partitions[1])[1]
    node_keys = [1:n_points;]

    # define maximum dimension for PH, at least 2
    max_dim = min(3, max_dim)

    # store all clusters that have been visited
    all_clusters = Any[]

    # store all 3-simplices and their birth times
    three_simplices = Any[]
    birth_three_simplices = Vector{Float64}()

    # store all 2-simplices and their birth times
    two_simplices = Any[]
    birth_two_simplices = Vector{Float64}()

    # store all 1-simplices and their birth times
    one_simplices = Any[]
    birth_one_simplices = Vector{Float64}()

    # store all 0-simplices and their birth times
    zero_simplices = Any[]
    birth_zero_simplices = Vector{Float64}()

    # store all simplices and birth times
    all_simplices = [zero_simplices, one_simplices, two_simplices, three_simplices]
    all_birth = [birth_zero_simplices, birth_one_simplices, birth_two_simplices, birth_three_simplices]

    ####################
    # obtain simplices #
    ####################

    println("Obtain simplices ... ")

    # iterate over all partitions #TODO include tqdm
    for m in ProgressBar(1:n_partitions)

        # get cluster id's and number of clusters
        cluster_id = partitions[m]
        min_cluster_id = minimum(cluster_id)
        max_cluster_id = maximum(cluster_id)

        for i in range(min_cluster_id, max_cluster_id)
            # get i-th cluster
            c = getindex(node_keys, cluster_id .== i)
            # only continue if cluster not empty
            if length(c) > 0

                # check if cluster was visited already (even as subset of other cluster)
                not_visited = true
                for visited_cluster in all_clusters
                    if intersect(c, visited_cluster) == c
                        not_visited = false
                        break
                    end
                end

                if not_visited
                    # add to visited clusters
                    push!(all_clusters, c)

                    # get size of cluster
                    s_c = size(c)[1]

                    # iterate through all sizes from min(cluseter size,max_dim+2) to 1
                    for k in reverse(1:min(s_c, max_dim + 1))
                        for s in Combinatorics.combinations(c, k)
                            # if k-1 simplex hasn't been visited
                            if !(s in all_simplices[k])
                                # add s to k-1 simplices
                                push!(all_simplices[k], s)
                                # store birth time
                                append!(all_birth[k], filtration_indices[m])
                            end
                        end
                    end
                end
            end
        end
    end

    ###########################
    # compute boundary matrix #
    ###########################

    println("Compute boundary matrix ... ")

    # compute number of simplices per dimension
    n_simplices_dim = [size(zero_simplices)[1], size(one_simplices)[1], size(two_simplices)[1], size(three_simplices)[1]]

    # compute cummulative sum
    n_simplices_cum = cumsum(n_simplices_dim)

    # compute total number of simplices
    n_simplices = sum(n_simplices_dim)

    # initialise boundary matrix
    # D = zeros(Int8, n_simplices, n_simplices)

    # initialise row indices
    rv = Vector{Int64}()

    # initialise cp
    cp = ones(Int64, n_simplices + 1)
    n_matrix_values = 1

    # compute faces of one simplices
    println("for dimension 0 ... ")
    for (j, s1) in ProgressBar(enumerate(one_simplices))
        for (i, s0) in enumerate(zero_simplices)
            if intersect(s0, s1) == s0
                # add to boundary matrix
                # D[i, j+n_simplices_cum[1]] = 1
                # store row index
                push!(rv, i)
                # increase number of matrix value by one
                n_matrix_values += 1
            end
        end
        cp[n_simplices_cum[1]+j+1] = n_matrix_values
    end

    # compute faces of two simplices
    println("for dimension 1 ... ")
    for (j, s2) in ProgressBar(enumerate(two_simplices))
        for (i, s1) in enumerate(one_simplices)
            if intersect(s1, s2) == s1
                # add to boundary matrix
                # D[i+n_simplices_cum[1], j+n_simplices_cum[2]] = 1
                # store row index
                push!(rv, i + n_simplices_cum[1])
                # increase number of matrix value by one
                n_matrix_values += 1
            end
        end
        cp[n_simplices_cum[2]+j+1] = n_matrix_values
    end

    # compute faces of three simplices
    println("for dimension 2 ... ")
    for (j, s3) in ProgressBar(enumerate(three_simplices))
        for (i, s2) in enumerate(two_simplices)
            if intersect(s2, s3) == s2
                # add to boundary matrix
                # D[i+n_simplices_cum[2], j+n_simplices_cum[3]] = 1
                # store row index
                push!(rv, i + n_simplices_cum[2])
                # increase number of matrix value by one
                n_matrix_values += 1
            end
        end
        cp[n_simplices_cum[3]+j+1] = n_matrix_values
    end

    ################################
    # collect birth and dimensions #
    ################################

    # collect birth times
    fv = zeros(n_simplices)
    fv[1:n_simplices_cum[1]] = birth_zero_simplices
    fv[n_simplices_cum[1]+1:n_simplices_cum[2]] = birth_one_simplices
    fv[n_simplices_cum[2]+1:n_simplices_cum[3]] = birth_two_simplices
    fv[n_simplices_cum[3]+1:n_simplices_cum[4]] = birth_three_simplices

    # store cell dimensions
    dv = zeros(Int64, n_simplices)
    dv[n_simplices_cum[1]+1:n_simplices_cum[2]] .= 1
    dv[n_simplices_cum[2]+1:n_simplices_cum[3]] .= 2
    dv[n_simplices_cum[3]+1:n_simplices_cum[4]] .= 3

    return rv, cp, dv, fv, all_simplices

end


function Compute_PH(rv=Vector{Int64}(), cp=Vector{Int64}(), dv=Vector{Int64}(), fv=Vector{Float64}(), all_simplices=Vector{Vector{Any}}(), max_dim=3)
    """Function computes PH with Eirene and collects PDs and cycle representatives."""
    # max_dim is at least 1
    max_dim = max(1, max_dim)

    # compute persistent homology with Eirene
    PH_Eirene = Eirene.eirene(rv=rv, cp=cp, dv=dv, fv=fv)

    # obtain 0-dim barcode
    PD0 = barcode_to_diagram(Eirene.barcode(PH_Eirene, dim=0))
    # compute all 0-dim class representatives
    CR0 = Any[]
    for i in range(1, size(PD0)[1])
        push!(CR0, process_classrep(Eirene.classrep(PH_Eirene, dim=0, class=i), all_simplices[1]))
    end

    # initialise
    PD1 = Vector{Tuple{Float64,Real}}()
    CR1 = Any[]
    if max_dim > 1
        # obtain 1-dim barcode
        PD1 = barcode_to_diagram(Eirene.barcode(PH_Eirene, dim=1))
        # compute all 1-dim class representatives
        if size(PD1)[1] > 0
            for i in range(1, size(PD1)[1])
                push!(CR1, process_classrep(Eirene.classrep(PH_Eirene, dim=1, class=i), all_simplices[2]))
            end
        end
    end

    # initialise
    PD2 = Vector{Tuple{Float64,Real}}()
    CR2 = Any[]
    if max_dim > 2
        # obtain 2-dim barcode
        PD2 = barcode_to_diagram(Eirene.barcode(PH_Eirene, dim=2))
        # compute all 2-dim class representatives
        if size(PD2)[1] > 0
            for i in range(1, size(PD2)[1])
                push!(CR2, process_classrep(Eirene.classrep(PH_Eirene, dim=2, class=i), all_simplices[3]))
            end
        end
    end


    return PD0, PD1, PD2, CR0, CR1, CR2, PH_Eirene
end

function barcode_to_diagram(barcode=Matrix{Float64}())
    """Function transforms Eirene barcode to input for PersistenceDiagrams.jl."""
    diagram = Vector{Tuple{Float64,Real}}()
    for i in range(1, size(barcode)[1])
        append!(diagram, [tuple(barcode[i, :][1], barcode[i, :][2])])
    end
    return diagram
end

function process_classrep(classrep, simplices)
    """Function replaces indices in classrep with simplices."""
    CR = Any[]
    for simplex_index in classrep
        push!(CR, simplices[simplex_index])
    end
    return CR
end