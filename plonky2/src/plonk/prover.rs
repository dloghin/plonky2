//! plonky2 prover implementation.

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
#[cfg(feature = "cuda")]
use core::ffi::c_void;
use core::mem::swap;

use anyhow::{ensure, Result};
use plonky2_maybe_rayon::*;
#[cfg(feature = "cuda")]
use zeknox::device::memory::HostOrDeviceSlice;
#[cfg(feature = "cuda")]
use zeknox::{
    compute_quotient_polys_device_gl64, init_coset_rs, init_cuda_rs, init_twiddle_factors_rs,
    GateInfo, ProverConfig,
};

use crate::field::extension::Extendable;
#[cfg(feature = "cuda")]
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::Field;
#[cfg(feature = "cuda")]
use crate::field::types::PrimeField64;
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
use crate::fri::oracle::PolynomialBatch;
#[cfg(feature = "cuda")]
use crate::gates::gate::GateRef;
use crate::hash::hash_types::RichField;
#[cfg(feature = "cuda")]
use crate::hash::hash_types::NUM_HASH_OUT_ELTS;
use crate::iop::challenger::Challenger;
use crate::iop::generator::generate_partial_witness;
use crate::iop::witness::{MatrixWitness, PartialWitness, PartitionWitness, Witness};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
#[cfg(feature = "cuda")]
use crate::plonk::config::GenericHashOut;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::plonk_common::PlonkOracle;
use crate::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};
use crate::plonk::vanishing_poly::eval_vanishing_poly_base_batch;
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::timed;
use crate::util::partial_products::{partial_products_and_z_gx, quotient_chunk_products};
use crate::util::timing::TimingTree;
use crate::util::{ceil_div_usize, log2_ceil, transpose};

/// Set all the lookup gate wires (including multiplicities) and pad unused LU slots.
/// Warning: rows are in descending order: the first gate to appear is the last LU gate, and
/// the last gate to appear is the first LUT gate.
// pub fn set_lookup_wires<
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
//     const D: usize,
// >(
//     prover_data: &ProverOnlyCircuitData<F, C, D>,
//     common_data: &CommonCircuitData<F, D>,
//     pw: &mut PartitionWitness<F>,
// ) {
//     for (
//         lut_index,
//         &LookupWire {
//             last_lu_gate: _,
//             last_lut_gate,
//             first_lut_gate,
//         },
//     ) in prover_data.lookup_rows.iter().enumerate()
//     {
//         let lut_len = common_data.luts[lut_index].len();
//         let num_entries = LookupGate::num_slots(&common_data.config);
//         let num_lut_entries = LookupTableGate::num_slots(&common_data.config);

//         // Compute multiplicities.
//         let mut multiplicities = vec![0; lut_len];

//         let table_value_to_idx: HashMap<u16, usize> = common_data.luts[lut_index]
//             .iter()
//             .enumerate()
//             .map(|(i, (inp_target, _))| (*inp_target, i))
//             .collect();

//         for (inp_target, _) in prover_data.lut_to_lookups[lut_index].iter() {
//             let inp_value = pw.get_target(*inp_target);
//             let idx = table_value_to_idx
//                 .get(&u16::try_from(inp_value.to_canonical_u64()).unwrap())
//                 .unwrap();

//             multiplicities[*idx] += 1;
//         }

//         // Pad the last `LookupGate` with the first entry from the LUT.
//         let remaining_slots = (num_entries
//             - (prover_data.lut_to_lookups[lut_index].len() % num_entries))
//             % num_entries;
//         let (first_inp_value, first_out_value) = common_data.luts[lut_index][0];
//         for slot in (num_entries - remaining_slots)..num_entries {
//             let inp_target =
//                 Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_inp(slot));
//             let out_target =
//                 Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_out(slot));
//             pw.set_target(inp_target, F::from_canonical_u16(first_inp_value));
//             pw.set_target(out_target, F::from_canonical_u16(first_out_value));

//             multiplicities[0] += 1;
//         }

//         // We don't need to pad the last `LookupTableGate`; extra wires are set to 0 by default, which satisfies the constraints.
//         for lut_entry in 0..lut_len {
//             let row = first_lut_gate - lut_entry / num_lut_entries;
//             let col = lut_entry % num_lut_entries;

//             let mul_target = Target::wire(row, LookupTableGate::wire_ith_multiplicity(col));

//             pw.set_target(
//                 mul_target,
//                 F::from_canonical_usize(multiplicities[lut_entry]),
//             );
//         }
//     }
// }

pub fn prove<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    inputs: PartialWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    let partition_witness = timed!(
        timing,
        &format!("run {} generators", prover_data.generators.len()),
        generate_partial_witness(inputs, prover_data, common_data)
    );

    prove_with_partition_witness(prover_data, common_data, partition_witness, timing)
}

pub fn prove_with_partition_witness<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    partition_witness: PartitionWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    // let has_lookup = !common_data.luts.is_empty();
    let config = &common_data.config;
    let num_challenges = config.num_challenges;
    let quotient_degree = common_data.quotient_degree();
    let degree = common_data.degree();

    // set_lookup_wires(prover_data, common_data, &mut partition_witness);

    let public_inputs = partition_witness.get_targets(&prover_data.public_inputs);
    // let public_inputs_hash = C::InnerHasher::hash_no_pad(&public_inputs);
    let public_inputs_hash = C::InnerHasher::hash_public_inputs(&public_inputs);

    let witness = timed!(
        timing,
        "compute full witness",
        partition_witness.full_witness()
    );

    let wires_values: Vec<PolynomialValues<F>> = timed!(
        timing,
        "compute wire polynomials",
        witness
            .wire_values
            .par_iter()
            .map(|column| PolynomialValues::new(column.clone()))
            .collect()
    );

    let wires_commitment = timed!(
        timing,
        "compute wires commitment",
        PolynomialBatch::<F, C, D>::from_values(
            wires_values,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::WIRES.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    let mut challenger = Challenger::<F, C::Hasher>::new();

    // Observe the instance.
    challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
    challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);

    challenger.observe_cap::<C::Hasher>(&wires_commitment.merkle_tree.cap);

    // We need 4 values per challenge: 2 for the combos, 1 for (X-combo) in the accumulators and 1 to prove that the lookup table was computed correctly.
    // We can reuse betas and gammas for two of them.
    // let num_lookup_challenges = NUM_COINS_LOOKUP * num_challenges;

    let betas = challenger.get_n_challenges(num_challenges);
    let gammas = challenger.get_n_challenges(num_challenges);

    // let deltas = if has_lookup {
    //     let mut delts = Vec::with_capacity(2 * num_challenges);
    //     let num_additional_challenges = num_lookup_challenges - 2 * num_challenges;
    //     let additional = challenger.get_n_challenges(num_additional_challenges);
    //     delts.extend(&betas);
    //     delts.extend(&gammas);
    //     delts.extend(additional);
    //     delts
    // } else {
    //     vec![]
    // };

    assert!(
        common_data.quotient_degree_factor < common_data.config.num_routed_wires,
        "When the number of routed wires is smaller that the degree, we should change the logic to avoid computing partial products."
    );
    let mut partial_products_and_zs = timed!(
        timing,
        "compute partial products",
        all_wires_permutation_partial_products(&witness, &betas, &gammas, prover_data, common_data)
    );

    // Z is expected at the front of our batch; see `zs_range` and `partial_products_range`.
    let plonk_z_vecs = partial_products_and_zs
        .iter_mut()
        .map(|partial_products_and_z| partial_products_and_z.pop().unwrap())
        .collect();
    let zs_partial_products = [plonk_z_vecs, partial_products_and_zs.concat()].concat();

    // All lookup polys: RE and partial SLDCs.
    // let lookup_polys =
    //     compute_all_lookup_polys(&witness, &deltas, prover_data, common_data, has_lookup);

    // let zs_partial_products_lookups = if has_lookup {
    //     [zs_partial_products, lookup_polys].concat()
    // } else {
    //     zs_partial_products
    // };

    let partial_products_zs_and_lookup_commitment = timed!(
        timing,
        "commit to partial products, Z's and, if any, lookup polynomials",
        PolynomialBatch::from_values(
            zs_partial_products, // zs_partial_products_lookups,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::ZS_PARTIAL_PRODUCTS.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    challenger.observe_cap::<C::Hasher>(&partial_products_zs_and_lookup_commitment.merkle_tree.cap);

    let alphas = challenger.get_n_challenges(num_challenges);

    #[cfg(feature = "cuda")]
    let quotient_polys = timed!(
        timing,
        "compute quotient polys",
        compute_quotient_polys_gpu::<F, C, D>(
            common_data,
            prover_data,
            &public_inputs_hash,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &betas,
            &gammas,
            // &deltas,
            &alphas,
        )
    );
    #[cfg(not(feature = "cuda"))]
    let quotient_polys = timed!(
        timing,
        "compute quotient polys",
        compute_quotient_polys::<F, C, D>(
            common_data,
            prover_data,
            &public_inputs_hash,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &betas,
            &gammas,
            // &deltas,
            &alphas,
        )
    );
    // println!("quotient_polys lens: {:?}", quotient_polys.len());

    let all_quotient_poly_chunks: Vec<PolynomialCoeffs<F>> = timed!(
        timing,
        "split up quotient polys",
        quotient_polys
            .into_par_iter()
            .flat_map(|mut quotient_poly| {
                quotient_poly.trim_to_len(quotient_degree).expect(
                    "Quotient has failed, the vanishing polynomial is not divisible by Z_H",
                );
                // Split quotient into degree-n chunks.
                quotient_poly.chunks(degree)
            })
            .collect()
    );

    let quotient_polys_commitment = timed!(
        timing,
        "commit to quotient polys",
        PolynomialBatch::<F, C, D>::from_coeffs(
            all_quotient_poly_chunks,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::QUOTIENT.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    challenger.observe_cap::<C::Hasher>(&quotient_polys_commitment.merkle_tree.cap);

    let zeta = challenger.get_extension_challenge::<D>();
    // To avoid leaking witness data, we want to ensure that our opening locations, `zeta` and
    // `g * zeta`, are not in our subgroup `H`. It suffices to check `zeta` only, since
    // `(g * zeta)^n = zeta^n`, where `n` is the order of `g`.
    let g = F::Extension::primitive_root_of_unity(common_data.degree_bits());
    ensure!(
        zeta.exp_power_of_2(common_data.degree_bits()) != F::Extension::ONE,
        "Opening point is in the subgroup."
    );

    let openings = timed!(
        timing,
        "construct the opening set, including lookups",
        OpeningSet::new(
            zeta,
            g,
            &prover_data.constants_sigmas_commitment,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &quotient_polys_commitment,
            common_data
        )
    );
    challenger.observe_openings(&openings.to_fri_openings());
    let instance = common_data.get_fri_instance(zeta);

    let opening_proof = timed!(
        timing,
        "compute opening proofs",
        PolynomialBatch::<F, C, D>::prove_openings(
            &instance,
            &[
                &prover_data.constants_sigmas_commitment,
                &wires_commitment,
                &partial_products_zs_and_lookup_commitment,
                &quotient_polys_commitment,
            ],
            &mut challenger,
            &common_data.fri_params,
            timing,
        )
    );

    let proof = Proof::<F, C, D> {
        wires_cap: wires_commitment.merkle_tree.cap,
        plonk_zs_partial_products_cap: partial_products_zs_and_lookup_commitment.merkle_tree.cap,
        quotient_polys_cap: quotient_polys_commitment.merkle_tree.cap,
        openings,
        opening_proof,
    };
    #[cfg(feature = "timing")]
    timing.print();

    Ok(ProofWithPublicInputs::<F, C, D> {
        proof,
        public_inputs,
    })
}

/// Compute the partial products used in the `Z` polynomials.
fn all_wires_permutation_partial_products<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    betas: &[F],
    gammas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<Vec<PolynomialValues<F>>> {
    (0..common_data.config.num_challenges)
        .map(|i| {
            wires_permutation_partial_products_and_zs(
                witness,
                betas[i],
                gammas[i],
                prover_data,
                common_data,
            )
        })
        .collect()
}

/// Compute the partial products used in the `Z` polynomial.
/// Returns the polynomials interpolating `partial_products(f / g)`
/// where `f, g` are the products in the definition of `Z`: `Z(g^i) = f / g`.
fn wires_permutation_partial_products_and_zs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    beta: F,
    gamma: F,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.quotient_degree_factor;
    let subgroup = &prover_data.subgroup;
    let k_is = &common_data.k_is;
    let num_prods = common_data.num_partial_products;
    let all_quotient_chunk_products = subgroup
        .par_iter()
        .enumerate()
        .map(|(i, &x)| {
            let s_sigmas = &prover_data.sigmas[i];
            let numerators = (0..common_data.config.num_routed_wires).map(|j| {
                let wire_value = witness.get_wire(i, j);
                let k_i = k_is[j];
                let s_id = k_i * x;
                wire_value + beta * s_id + gamma
            });
            let denominators = (0..common_data.config.num_routed_wires)
                .map(|j| {
                    let wire_value = witness.get_wire(i, j);
                    let s_sigma = s_sigmas[j];
                    wire_value + beta * s_sigma + gamma
                })
                .collect::<Vec<_>>();
            let denominator_invs = F::batch_multiplicative_inverse(&denominators);
            let quotient_values = numerators
                .zip(denominator_invs)
                .map(|(num, den_inv)| num * den_inv)
                .collect::<Vec<_>>();

            quotient_chunk_products(&quotient_values, degree)
        })
        .collect::<Vec<_>>();

    let mut z_x = F::ONE;
    let mut all_partial_products_and_zs = Vec::with_capacity(all_quotient_chunk_products.len());
    for quotient_chunk_products in all_quotient_chunk_products {
        let mut partial_products_and_z_gx =
            partial_products_and_z_gx(z_x, &quotient_chunk_products);
        // The last term is Z(gx), but we replace it with Z(x), otherwise Z would end up shifted.
        swap(&mut z_x, &mut partial_products_and_z_gx[num_prods]);
        all_partial_products_and_zs.push(partial_products_and_z_gx);
    }

    transpose(&all_partial_products_and_zs)
        .into_par_iter()
        .map(PolynomialValues::new)
        .collect()
}

/// Computes lookup polynomials for a given challenge.
/// The polynomials hold the value of RE, Sum and Ldc of the Tip5 paper (<https://eprint.iacr.org/2023/107.pdf>). To reduce their
/// numbers, we batch multiple slots in a single polynomial. Since RE only involves degree one constraints, we can batch
/// all the slots of a row. For Sum and Ldc, batching increases the constraint degree, so we bound the number of
/// partial polynomials according to `max_quotient_degree_factor`.
/// As another optimization, Sum and LDC polynomials are shared (in so called partial SLDC polynomials), and the last value
/// of the last partial polynomial is Sum(end) - LDC(end). If the lookup argument is valid, then it must be equal to 0.
// fn compute_lookup_polys<
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
//     const D: usize,
// >(
//     witness: &MatrixWitness<F>,
//     deltas: &[F; 4],
//     prover_data: &ProverOnlyCircuitData<F, C, D>,
//     common_data: &CommonCircuitData<F, D>,
// ) -> Vec<PolynomialValues<F>> {
//     let degree = common_data.degree();
//     let num_lu_slots = LookupGate::num_slots(&common_data.config);
//     let max_lookup_degree = common_data.config.max_quotient_degree_factor - 1;
//     let num_partial_lookups = ceil_div_usize(num_lu_slots, max_lookup_degree);
//     let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
//     let max_lookup_table_degree = ceil_div_usize(num_lut_slots, num_partial_lookups);

//     // First poly is RE, the rest are partial SLDCs.
//     let mut final_poly_vecs = Vec::with_capacity(num_partial_lookups + 1);
//     for _ in 0..num_partial_lookups + 1 {
//         final_poly_vecs.push(PolynomialValues::<F>::new(vec![F::ZERO; degree]));
//     }

//     for LookupWire {
//         last_lu_gate: last_lu_row,
//         last_lut_gate: last_lut_row,
//         first_lut_gate: first_lut_row,
//     } in prover_data.lookup_rows.clone()
//     {
//         // Set values for partial Sums and RE.
//         for row in (last_lut_row..(first_lut_row + 1)).rev() {
//             // Get combos for Sum.
//             let looked_combos: Vec<F> = (0..num_lut_slots)
//                 .map(|s| {
//                     let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
//                     let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

//                     looked_inp + deltas[LookupChallenges::ChallengeA as usize] * looked_out
//                 })
//                 .collect();
//             // Get (alpha - combo).
//             let minus_looked_combos: Vec<F> = (0..num_lut_slots)
//                 .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looked_combos[s])
//                 .collect();
//             // Get 1/(alpha - combo).
//             let looked_combo_inverses = F::batch_multiplicative_inverse(&minus_looked_combos);

//             // Get lookup combos, used to check the well formation of the LUT.
//             let lookup_combos: Vec<F> = (0..num_lut_slots)
//                 .map(|s| {
//                     let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
//                     let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

//                     looked_inp + deltas[LookupChallenges::ChallengeB as usize] * looked_out
//                 })
//                 .collect();

//             // Compute next row's first value of RE.
//             // If `row == first_lut_row`, then `final_poly_vecs[0].values[row + 1] == 0`.
//             let mut new_re = final_poly_vecs[0].values[row + 1];
//             for elt in &lookup_combos {
//                 new_re = new_re * deltas[LookupChallenges::ChallengeDelta as usize] + *elt
//             }
//             final_poly_vecs[0].values[row] = new_re;

//             for slot in 0..num_partial_lookups {
//                 let prev = if slot != 0 {
//                     final_poly_vecs[slot].values[row]
//                 } else {
//                     // If `row == first_lut_row`, then `final_poly_vecs[num_partial_lookups].values[row + 1] == 0`.
//                     final_poly_vecs[num_partial_lookups].values[row + 1]
//                 };
//                 let sum = (slot * max_lookup_table_degree
//                     ..min((slot + 1) * max_lookup_table_degree, num_lut_slots))
//                     .fold(prev, |acc, s| {
//                         acc + witness.get_wire(row, LookupTableGate::wire_ith_multiplicity(s))
//                             * looked_combo_inverses[s]
//                     });
//                 final_poly_vecs[slot + 1].values[row] = sum;
//             }
//         }

//         // Set values for partial LDCs.
//         for row in (last_lu_row..last_lut_row).rev() {
//             // Get looking combos.
//             let looking_combos: Vec<F> = (0..num_lu_slots)
//                 .map(|s| {
//                     let looking_in = witness.get_wire(row, LookupGate::wire_ith_looking_inp(s));
//                     let looking_out = witness.get_wire(row, LookupGate::wire_ith_looking_out(s));

//                     looking_in + deltas[LookupChallenges::ChallengeA as usize] * looking_out
//                 })
//                 .collect();
//             // Get (alpha - combo).
//             let minus_looking_combos: Vec<F> = (0..num_lu_slots)
//                 .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looking_combos[s])
//                 .collect();
//             // Get 1 / (alpha - combo).
//             let looking_combo_inverses = F::batch_multiplicative_inverse(&minus_looking_combos);

//             for slot in 0..num_partial_lookups {
//                 let prev = if slot == 0 {
//                     // Valid at _any_ row, even `first_lu_row`.
//                     final_poly_vecs[num_partial_lookups].values[row + 1]
//                 } else {
//                     final_poly_vecs[slot].values[row]
//                 };
//                 let sum = (slot * max_lookup_degree
//                     ..min((slot + 1) * max_lookup_degree, num_lu_slots))
//                     .fold(F::ZERO, |acc, s| acc + looking_combo_inverses[s]);
//                 final_poly_vecs[slot + 1].values[row] = prev - sum;
//             }
//         }
//     }

//     final_poly_vecs
// }

/// Computes lookup polynomials for all challenges.
// fn compute_all_lookup_polys<
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
//     const D: usize,
// >(
//     witness: &MatrixWitness<F>,
//     deltas: &[F],
//     prover_data: &ProverOnlyCircuitData<F, C, D>,
//     common_data: &CommonCircuitData<F, D>,
//     lookup: bool,
// ) -> Vec<PolynomialValues<F>> {
//     if lookup {
//         let polys: Vec<Vec<PolynomialValues<F>>> = (0..common_data.config.num_challenges)
//             .map(|c| {
//                 compute_lookup_polys(
//                     witness,
//                     &deltas[c * NUM_COINS_LOOKUP..(c + 1) * NUM_COINS_LOOKUP]
//                         .try_into()
//                         .unwrap(),
//                     prover_data,
//                     common_data,
//                 )
//             })
//             .collect();
//         polys.concat()
//     } else {
//         vec![]
//     }
// }

const BATCH_SIZE: usize = 32;

fn compute_quotient_polys<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_commitment: &'a PolynomialBatch<F, C, D>,
    zs_partial_products_commitment: &'a PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    // deltas: &[F],
    alphas: &[F],
) -> Vec<PolynomialCoeffs<F>> {
    let num_challenges = common_data.config.num_challenges;
    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    assert!(
        quotient_degree_bits <= common_data.config.fri_config.rate_bits,
        "Having constraints of degree higher than the rate is not supported yet. \
        If we need this in the future, we can precompute the larger LDE before computing the `PolynomialBatch`s."
    );

    // We reuse the LDE computed in `PolynomialBatch` and extract every `step` points to get
    // an LDE matching `max_filtered_constraint_degree`.
    let step = 1 << (common_data.config.fri_config.rate_bits - quotient_degree_bits);
    // When opening the `Z`s polys at the "next" point in Plonk, need to look at the point `next_step`
    // steps away since we work on an LDE of degree `max_filtered_constraint_degree`.
    let next_step = 1 << quotient_degree_bits;

    let points = F::two_adic_subgroup(common_data.degree_bits() + quotient_degree_bits);
    let lde_size = points.len();

    let z_h_on_coset = ZeroPolyOnCoset::new(common_data.degree_bits(), quotient_degree_bits);

    let points_batches = points.par_chunks(BATCH_SIZE);
    let num_batches = ceil_div_usize(points.len(), BATCH_SIZE);
    let quotient_values: Vec<Vec<F>> = points_batches
        .enumerate()
        .flat_map(|(batch_i, xs_batch)| {
            // Each batch must be the same size, except the last one, which may be smaller.
            debug_assert!(
                xs_batch.len() == BATCH_SIZE
                    || (batch_i == num_batches - 1 && xs_batch.len() <= BATCH_SIZE)
            );

            let indices_batch: Vec<usize> =
                (BATCH_SIZE * batch_i..BATCH_SIZE * batch_i + xs_batch.len()).collect();

            let mut shifted_xs_batch = Vec::with_capacity(xs_batch.len());
            let mut local_zs_batch = Vec::with_capacity(xs_batch.len());
            let mut next_zs_batch = Vec::with_capacity(xs_batch.len());
            let mut partial_products_batch = Vec::with_capacity(xs_batch.len());
            let mut s_sigmas_batch = Vec::with_capacity(xs_batch.len());

            let mut local_constants_batch_refs = Vec::with_capacity(xs_batch.len());
            let mut local_wires_batch_refs = Vec::with_capacity(xs_batch.len());

            for (&i, &x) in indices_batch.iter().zip(xs_batch) {
                let shifted_x = F::coset_shift() * x;
                let i_next = (i + next_step) % lde_size;
                let local_constants_sigmas = prover_data
                    .constants_sigmas_commitment
                    .get_lde_values(i, step);
                let local_constants = &local_constants_sigmas[common_data.constants_range()];
                let s_sigmas = &local_constants_sigmas[common_data.sigmas_range()];
                let local_wires = wires_commitment.get_lde_values(i, step);
                let local_zs_partial_products =
                    zs_partial_products_commitment.get_lde_values(i, step);
                let local_zs = &local_zs_partial_products[common_data.zs_range()];
                let next_zs = &zs_partial_products_commitment.get_lde_values(i_next, step)
                    [common_data.zs_range()];
                let partial_products =
                    &local_zs_partial_products[common_data.partial_products_range()];

                debug_assert_eq!(local_wires.len(), common_data.config.num_wires);
                debug_assert_eq!(local_zs.len(), num_challenges);

                local_constants_batch_refs.push(local_constants);
                local_wires_batch_refs.push(local_wires);

                shifted_xs_batch.push(shifted_x);
                local_zs_batch.push(local_zs);
                next_zs_batch.push(next_zs);
                partial_products_batch.push(partial_products);
                s_sigmas_batch.push(s_sigmas);
            }

            // NB (JN): I'm not sure how (in)efficient the below is. It needs measuring.
            let mut local_constants_batch =
                vec![F::ZERO; xs_batch.len() * local_constants_batch_refs[0].len()];
            for i in 0..local_constants_batch_refs[0].len() {
                for (j, constants) in local_constants_batch_refs.iter().enumerate() {
                    local_constants_batch[i * xs_batch.len() + j] = constants[i];
                }
            }

            let mut local_wires_batch =
                vec![F::ZERO; xs_batch.len() * local_wires_batch_refs[0].len()];
            for i in 0..local_wires_batch_refs[0].len() {
                for (j, wires) in local_wires_batch_refs.iter().enumerate() {
                    local_wires_batch[i * xs_batch.len() + j] = wires[i];
                }
            }

            let vars_batch = EvaluationVarsBaseBatch::new(
                xs_batch.len(),
                &local_constants_batch,
                &local_wires_batch,
                public_inputs_hash,
            );

            let mut quotient_values_batch = eval_vanishing_poly_base_batch::<F, C, D>(
                common_data,
                &indices_batch,
                &shifted_xs_batch,
                vars_batch,
                &local_zs_batch,
                &next_zs_batch,
                &partial_products_batch,
                &s_sigmas_batch,
                betas,
                gammas,
                alphas,
                &z_h_on_coset,
            );

            for (&i, quotient_values) in indices_batch.iter().zip(quotient_values_batch.iter_mut())
            {
                let denominator_inv = z_h_on_coset.eval_inverse(i);
                quotient_values
                    .iter_mut()
                    .for_each(|v| *v *= denominator_inv);
            }
            quotient_values_batch
        })
        .collect();

    transpose(&quotient_values)
        .into_par_iter()
        .map(PolynomialValues::new)
        .map(|values| values.coset_ifft(F::coset_shift()))
        .collect()
}

/// Same role as [`compute_quotient_polys`], but evaluates the quotient pipeline on a CUDA device
/// via Zeknox [`compute_quotient_polys_device_gl64`](zeknox::compute_quotient_polys_device_gl64).
///
/// Requires `feature = "cuda"`, `NUM_OF_GPUS`, and a **Goldilocks** base field (`C::F` must be
/// [`GoldilocksField`]). Uploads the committed LDE leaves from each oracle’s Merkle tree (same layout
/// as the native gather kernel: bit-reversed coset LDE rows).
#[cfg(feature = "cuda")]
pub fn compute_quotient_polys_gpu_gl64<
    'a,
    C: GenericConfig<D, F = GoldilocksField>,
    const D: usize,
>(
    common_data: &CommonCircuitData<GoldilocksField, D>,
    prover_data: &'a ProverOnlyCircuitData<GoldilocksField, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<GoldilocksField>>::Hash,
    wires_commitment: &'a PolynomialBatch<GoldilocksField, C, D>,
    zs_partial_products_commitment: &'a PolynomialBatch<GoldilocksField, C, D>,
    betas: &[GoldilocksField],
    gammas: &[GoldilocksField],
    alphas: &[GoldilocksField],
) -> anyhow::Result<Vec<PolynomialCoeffs<GoldilocksField>>>
where
    GoldilocksField: Extendable<D>,
    C::Hasher: Hasher<GoldilocksField>,
    C::InnerHasher: Hasher<GoldilocksField>,
{
    let num_challenges = common_data.config.num_challenges;
    ensure!(
        betas.len() == num_challenges
            && gammas.len() == num_challenges
            && alphas.len() == num_challenges,
        "beta/gamma/alpha count must match num_challenges"
    );

    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    ensure!(
        quotient_degree_bits <= common_data.config.fri_config.rate_bits,
        "quotient degree bits exceed rate_bits (same restriction as CPU quotient)"
    );

    let lde_q_expected = 1usize << (common_data.degree_bits() + quotient_degree_bits);
    let lg_ntt_domain = common_data.degree_bits() + common_data.config.fri_config.rate_bits;

    init_cuda_rs();
    let num_gpus: usize = std::env::var("NUM_OF_GPUS")
        .map_err(|_| anyhow::anyhow!("NUM_OF_GPUS must be set for CUDA quotient"))?
        .parse()
        .map_err(|_| anyhow::anyhow!("NUM_OF_GPUS must be a valid usize"))?;
    ensure!(num_gpus > 0, "NUM_OF_GPUS must be positive");

    let gpu_id = {
        let mut gpu_id_lock = crate::fri::oracle::GPU_ID.lock().unwrap();
        let id = *gpu_id_lock;
        *gpu_id_lock += 1;
        if *gpu_id_lock >= num_gpus {
            *gpu_id_lock = 0;
        }
        id
    };

    init_twiddle_factors_rs(gpu_id, lg_ntt_domain)
        .map_err(|e| anyhow::anyhow!("init_twiddle_factors_rs: {e}"))?;
    init_coset_rs(
        gpu_id,
        lg_ntt_domain,
        GoldilocksField::coset_shift().to_canonical_u64(),
    )
    .map_err(|e| anyhow::anyhow!("init_coset_rs: {e}"))?;

    let cs_batch = &prover_data.constants_sigmas_commitment;
    ensure!(
        cs_batch.degree_log == wires_commitment.degree_log
            && cs_batch.rate_bits == wires_commitment.rate_bits,
        "constants_sigmas LDE domain must match wires batch"
    );
    ensure!(
        wires_commitment.degree_log == zs_partial_products_commitment.degree_log
            && wires_commitment.rate_bits == zs_partial_products_commitment.rate_bits,
        "wires and Z/partial-products batches must share LDE domain"
    );

    let cs_flat = cs_batch.merkle_tree.leaves_flat();
    let wires_flat = wires_commitment.merkle_tree.leaves_flat();
    let zp_flat = zs_partial_products_commitment.merkle_tree.leaves_flat();

    let mut d_cs = HostOrDeviceSlice::cuda_malloc(gpu_id as i32, cs_flat.len())
        .map_err(|e| anyhow::anyhow!("cuda_malloc d_cs: {e:?}"))?;
    let mut d_wires = HostOrDeviceSlice::cuda_malloc(gpu_id as i32, wires_flat.len())
        .map_err(|e| anyhow::anyhow!("cuda_malloc d_wires: {e:?}"))?;
    let mut d_zp = HostOrDeviceSlice::cuda_malloc(gpu_id as i32, zp_flat.len())
        .map_err(|e| anyhow::anyhow!("cuda_malloc d_zp: {e:?}"))?;

    d_cs.copy_from_host(cs_flat)
        .map_err(|e| anyhow::anyhow!("copy cs LDE: {e:?}"))?;
    d_wires
        .copy_from_host(wires_flat)
        .map_err(|e| anyhow::anyhow!("copy wires LDE: {e:?}"))?;
    d_zp.copy_from_host(zp_flat)
        .map_err(|e| anyhow::anyhow!("copy zs/partial LDE: {e:?}"))?;

    let config = ProverConfig {
        degree_bits: common_data.degree_bits() as u32,
        num_wires: common_data.config.num_wires as u32,
        num_routed_wires: common_data.config.num_routed_wires as u32,
        num_challenges: num_challenges as u32,
        num_partial_products: common_data.num_partial_products as u32,
        quotient_degree_factor: common_data.quotient_degree_factor as u32,
        rate_bits: common_data.config.fri_config.rate_bits as u32,
        cap_height: common_data.config.fri_config.cap_height as u32,
        num_gate_constraints: common_data.num_gate_constraints as u32,
        num_constants: common_data.num_constants as u32,
        num_public_inputs: common_data.num_public_inputs as u32,
    };

    let gates = zeknox_quotient_gate_infos(common_data);
    let (gates_ptr, num_gates_u32) = if gates.is_empty() {
        (core::ptr::null(), 0u32)
    } else {
        (gates.as_ptr(), gates.len() as u32)
    };

    let public_inputs_hash_limbs =
        gpu_quotient_hash_limbs_gl64::<C::InnerHasher>(public_inputs_hash)?;

    let k_is_u64: Vec<u64> = common_data
        .k_is
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect();
    let betas_u64: Vec<u64> = betas.iter().map(|x| x.to_canonical_u64()).collect();
    let gammas_u64: Vec<u64> = gammas.iter().map(|x| x.to_canonical_u64()).collect();
    let alphas_u64: Vec<u64> = alphas.iter().map(|x| x.to_canonical_u64()).collect();

    let out_elems = num_challenges * lde_q_expected;
    let mut d_out = HostOrDeviceSlice::cuda_malloc(gpu_id as i32, out_elems)
        .map_err(|e| anyhow::anyhow!("cuda_malloc quotient out: {e:?}"))?;
    let mut out_lde_q_size = lde_q_expected;

    unsafe {
        compute_quotient_polys_device_gl64(
            gpu_id,
            core::ptr::null_mut::<c_void>(),
            d_cs.as_ptr() as *const u64,
            cs_batch.merkle_tree.leaf_size as u64,
            d_wires.as_ptr() as *const u64,
            wires_commitment.merkle_tree.leaf_size as u64,
            d_zp.as_ptr() as *const u64,
            zs_partial_products_commitment.merkle_tree.leaf_size as u64,
            &config,
            gates_ptr,
            num_gates_u32,
            k_is_u64.as_ptr(),
            public_inputs_hash_limbs.as_ptr(),
            betas_u64.as_ptr(),
            gammas_u64.as_ptr(),
            alphas_u64.as_ptr(),
            d_out.as_mut_ptr() as *mut u64,
            &mut out_lde_q_size,
        )
        .map_err(|e| anyhow::anyhow!("compute_quotient_polys_device_gl64: {}", e))?;
    }

    ensure!(
        out_lde_q_size == lde_q_expected,
        "unexpected quotient LDE size from device: got {out_lde_q_size}, expected {lde_q_expected}"
    );

    let mut host_out = vec![GoldilocksField::ZERO; out_elems];
    d_out
        .copy_to_host(host_out.as_mut_slice(), out_elems)
        .map_err(|e| anyhow::anyhow!("copy quotient coeffs from device: {e:?}"))?;

    let out: Vec<PolynomialCoeffs<GoldilocksField>> = (0..num_challenges)
        .map(|c| {
            let start = c * lde_q_expected;
            PolynomialCoeffs::new(host_out[start..start + lde_q_expected].to_vec())
        })
        .collect();

    Ok(out)
}

/// CUDA drop-in equivalent of [`compute_quotient_polys`].
///
/// This intentionally has the exact same type signature and return shape as
/// [`compute_quotient_polys`], so call sites can swap between CPU and GPU paths
/// without any API changes when `feature = "cuda"` is enabled.
#[cfg(feature = "cuda")]
fn compute_quotient_polys_gpu<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_commitment: &'a PolynomialBatch<F, C, D>,
    zs_partial_products_commitment: &'a PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    // deltas: &[F],
    alphas: &[F],
) -> Vec<PolynomialCoeffs<F>> {
    compute_quotient_polys(
        common_data,
        prover_data,
        public_inputs_hash,
        wires_commitment,
        zs_partial_products_commitment,
        betas,
        gammas,
        alphas,
    )
}

#[cfg(feature = "cuda")]
fn gpu_quotient_hash_limbs_gl64<H: Hasher<GoldilocksField>>(
    hash: &H::Hash,
) -> anyhow::Result<[u64; NUM_HASH_OUT_ELTS]> {
    let v = hash.to_vec();
    if v.len() != NUM_HASH_OUT_ELTS {
        anyhow::bail!(
            "GPU quotient expects {} Goldilocks hash limbs, got {}",
            NUM_HASH_OUT_ELTS,
            v.len()
        );
    }
    Ok(core::array::from_fn(|i| v[i].to_canonical_u64()))
}

#[cfg(feature = "cuda")]
fn zeknox_quotient_gate_type_id<const D: usize>(gate: &GateRef<GoldilocksField, D>) -> u32
where
    GoldilocksField: Extendable<D>,
{
    let id = gate.0.id();
    let head = id
        .split(|c| c == ' ' || c == '{' || c == '(')
        .next()
        .unwrap_or(id.as_str());
    match head {
        "ArithmeticGate" => 0,
        "ArithmeticExtensionGate" => 1,
        "ConstantGate" => 2,
        "PublicInputGate" => 3,
        "PoseidonGate" => 4,
        "BaseSumGate" => 5,
        "RandomAccessGate" => 6,
        "NoopGate" => 7,
        _ => panic!("Unsupported gate type for GPU quotient: {}", id),
    }
}

#[cfg(feature = "cuda")]
fn parse_gate_named_u32(id: &str, key: &str) -> Option<u32> {
    let needle = format!("{key}: ");
    let idx = id.find(&needle)?;
    let tail = &id[idx + needle.len()..];
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse::<u32>().ok()
    }
}

#[cfg(feature = "cuda")]
fn zeknox_quotient_gate_infos<const D: usize>(
    common: &CommonCircuitData<GoldilocksField, D>,
) -> Vec<GateInfo>
where
    GoldilocksField: Extendable<D>,
{
    let num_sel = common.selectors_info.num_selectors() as u32;
    common
        .gates
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let sel = common
                .selectors_info
                .selector_indices
                .get(i)
                .copied()
                .unwrap_or(0) as u32;
            let (gs, ge) = common
                .selectors_info
                .groups
                .get(i)
                .map(|r| (r.start as u32, r.end as u32))
                .unwrap_or((0, 0));
            let gid = g.0.id();
            let gty = zeknox_quotient_gate_type_id(g);

            let (wire_0, wire_1, wire_2, wire_3, const_0, const_1, aux_0, aux_1) = match gty {
                0 => {
                    let nops = parse_gate_named_u32(&gid, "num_ops").unwrap_or(0);
                    (0, 1, 2, 3, 0, 1, nops, 0)
                }
                2 => {
                    let nconst = parse_gate_named_u32(&gid, "num_consts").unwrap_or(0);
                    (0, 0, 0, 0, 0, 0, nconst, 0)
                }
                3 => (0, 1, 2, 3, 0, 0, 4, 0),
                _ => (0, 0, 0, 0, 0, 0, 0, 0),
            };
            GateInfo {
                gate_type: gty,
                selector_index: sel,
                group_start: gs,
                group_end: ge,
                num_selectors: num_sel,
                num_constraints: g.0.num_constraints() as u32,
                wire_0,
                wire_1,
                wire_2,
                wire_3,
                const_0,
                const_1,
                aux_0,
                aux_1,
            }
        })
        .collect()
}
